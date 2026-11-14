"""Fine-tune and evaluate answer generators on the context-level split.

  python gen_eval.py --model t5    # google-t5/t5-base, full fine-tuning
  python gen_eval.py --model phi2  # microsoft/phi-2 + LoRA, loss on answer tokens only

Training uses evaluation/splits/train.json only; the epoch is selected on val.json;
test.json is scored once. Test settings: gold (oracle) context, and the top-1 / top-3
contexts returned by the hybrid retriever (end-to-end RAG).
"""
import argparse
import json
import math
import random
import time

import numpy as np
import torch
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU

from common import RESULTS, SEED, bootstrap_ci, exact_match, f1_score, load_split, load_squad, normalize_answer

torch.manual_seed(SEED)
random.seed(SEED)
torch.set_num_threads(8)


def examples(split):
    return [{"id": q["id"], "cid": c["cid"], "context": c["context"], "question": q["question"],
             "answer": q["answer"]} for c in split for q in c["qas"]]


class T5:
    name = "google-t5/t5-base"

    def __init__(self, max_len):
        from transformers import AutoTokenizer, T5ForConditionalGeneration
        self.tok = AutoTokenizer.from_pretrained(self.name)
        self.model = T5ForConditionalGeneration.from_pretrained(self.name)
        self.max_len = max_len

    def params(self):
        return self.model.parameters()

    def _src(self, b):
        return [f"question: {x['question']} context: {x['context']}" for x in b]

    def loss(self, b):
        enc = self.tok(self._src(b), max_length=self.max_len, truncation=True, padding=True, return_tensors="pt")
        lab = self.tok([x["answer"] for x in b], max_length=32, truncation=True, padding=True, return_tensors="pt").input_ids
        lab[lab == self.tok.pad_token_id] = -100
        return self.model(**enc, labels=lab).loss

    @torch.no_grad()
    def predict(self, b):
        enc = self.tok(self._src(b), max_length=self.max_len, truncation=True, padding=True, return_tensors="pt")
        out = self.model.generate(**enc, max_new_tokens=32, num_beams=1)
        return self.tok.batch_decode(out, skip_special_tokens=True)


class Phi2:
    name = "microsoft/phi-2"

    def __init__(self, max_len):
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(self.name)
        self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"
        base = AutoModelForCausalLM.from_pretrained(self.name, torch_dtype=torch.float32)
        cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.1, bias="none", task_type="CAUSAL_LM",
                         target_modules=["q_proj", "k_proj", "v_proj", "dense"])
        self.model = get_peft_model(base, cfg)
        for p in self.model.parameters():
            if p.requires_grad:
                p.data = p.data.float()
        self.max_len = max_len

    def params(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def prompt_ids(self, x, budget):
        """Truncate the context, never the question/'Answer:' suffix, to fit `budget` tokens."""
        suffix = self.tok(f"\nQuestion: {x['question']}\nAnswer:").input_ids
        return self.tok(x["context"]).input_ids[:max(budget - len(suffix), 0)] + suffix

    def loss(self, b):
        ids, labels = [], []
        for x in b:
            a = self.tok(" " + x["answer"]).input_ids + [self.tok.eos_token_id]
            p = self.prompt_ids(x, self.max_len - len(a))
            ids.append(p + a)
            labels.append([-100] * len(p) + a)  # prompt masked: loss only on answer + EOS
        n = max(map(len, ids))
        pad = self.tok.eos_token_id
        inp = torch.tensor([[pad] * (n - len(i)) + i for i in ids])
        lab = torch.tensor([[-100] * (n - len(l)) + l for l in labels])
        att = torch.tensor([[0] * (n - len(i)) + [1] * len(i) for i in ids])
        return self.model(input_ids=inp, attention_mask=att, labels=lab).loss

    @torch.no_grad()
    def predict(self, b):
        ids = [self.prompt_ids(x, self.max_len - 24) for x in b]
        n = max(map(len, ids))
        pad = self.tok.eos_token_id
        enc = {"input_ids": torch.tensor([[pad] * (n - len(i)) + i for i in ids]),
               "attention_mask": torch.tensor([[0] * (n - len(i)) + [1] * len(i) for i in ids])}
        out = self.model.generate(**enc, max_new_tokens=24, do_sample=False, pad_token_id=self.tok.eos_token_id)
        text = self.tok.batch_decode(out[:, n:], skip_special_tokens=True)
        return [t.strip().split("\n")[0].strip() for t in text]


def run_predict(m, data, bs):
    m.model.eval()
    preds = []
    for i in range(0, len(data), bs):
        preds += m.predict(data[i:i + bs])
    return preds


def score(data, preds):
    em = [exact_match(p, x["answer"]) for p, x in zip(preds, data)]
    f1 = [f1_score(p, x["answer"]) for p, x in zip(preds, data)]
    groups = [x["cid"] for x in data]
    rs = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    rl = [rs.score(x["answer"], p)["rougeL"].fmeasure for p, x in zip(preds, data)]
    bleu = BLEU(effective_order=True).corpus_score(preds, [[x["answer"] for x in data]]).score / 100
    return {"n_questions": len(data), "n_contexts": len(set(groups)),
            "exact_match": float(np.mean(em)), "exact_match_95ci": bootstrap_ci(em, groups),
            "f1": float(np.mean(f1)), "f1_95ci": bootstrap_ci(f1, groups),
            "rougeL": float(np.mean(rl)), "rougeL_95ci": bootstrap_ci(rl, groups),
            "bleu": bleu}


def with_retrieved(data, retrieved, ctx_text, k):
    return [{**x, "context": "\n\n".join(ctx_text[c] for c in retrieved[x["id"]][:k])} for x in data]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["t5", "phi2"], required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--settings", default="gold,top1,top3")
    ap.add_argument("--no_val", action="store_true", help="skip per-epoch validation (single-epoch runs)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--max_per_fact", type=int, default=0,
                    help="keep at most N paraphrased training questions per (context, answer) fact; 0 = all")
    args = ap.parse_args()
    lr = args.lr or (3e-4 if args.model == "t5" else 2e-4)
    m = (T5 if args.model == "t5" else Phi2)(args.max_len)
    train, val, test = (examples(load_split(s)) for s in ("train", "val", "test"))
    if args.max_per_fact:
        seen, kept = {}, []
        for x in random.Random(SEED).sample(train, len(train)):
            key = (x["cid"], normalize_answer(x["answer"]))
            seen[key] = seen.get(key, 0) + 1
            if seen[key] <= args.max_per_fact:
                kept.append(x)
        train = kept
    opt = torch.optim.AdamW(m.params(), lr=lr, weight_decay=0.01)
    steps = args.epochs * math.ceil(len(train) / args.bs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1 - s / max(steps, 1)))
    log, best, best_state = [], -1, None
    rng = random.Random(SEED)
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        m.model.train()
        order = list(train)
        rng.shuffle(order)
        losses = []
        for i in range(0, len(order), args.bs):
            loss = m.loss(order[i:i + args.bs])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.params(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            losses.append(loss.item())
            if len(losses) % 20 == 0:
                print(f"ep{ep} step{len(losses)} loss {np.mean(losses[-20:]):.4f} {time.time() - t0:.0f}s", flush=True)
        v = {"exact_match": float("nan"), "f1": float("nan")} if args.no_val else score(val, run_predict(m, val, args.bs * 2))
        log.append({"epoch": ep, "train_loss": float(np.mean(losses)), "val_em": v["exact_match"], "val_f1": v["f1"]})
        print(log[-1], flush=True)
        if args.no_val or v["exact_match"] > best:
            best = v["exact_match"]
            best_state = {k: t.detach().clone() for k, t in m.model.state_dict().items()
                          if args.model == "t5" or "lora" in k}
    if best_state is not None:
        m.model.load_state_dict(best_state, strict=False)
    ctx_text = {c["cid"]: c["context"] for c in load_squad()}
    retrieved = json.load(open(RESULTS / "retrieved_contexts_hybrid.json"))["test"]
    res = {"model": m.name, "lr": lr, "epochs": args.epochs, "batch_size": args.bs, "max_per_fact": args.max_per_fact,
           "train_questions": len(train), "val_questions": len(val), "selected_epoch":
           (log[-1]["epoch"] if args.no_val else max(log, key=lambda r: r["val_em"])["epoch"]) if log else 0, "training_log": log, "test": {}}
    preds_out = {}
    for s in args.settings.split(","):
        data = test if s == "gold" else with_retrieved(test, retrieved, ctx_text, int(s[3:]))
        preds = run_predict(m, data, args.bs * 2)
        res["test"][s] = score(data, preds)
        preds_out[s] = [{"id": x["id"], "question": x["question"], "gold": x["answer"], "pred": p}
                        for x, p in zip(data, preds)]
        print(s, res["test"][s], flush=True)
    res["wall_time_s"] = time.time() - t0
    json.dump(res, open(RESULTS / f"generation_{args.model}{args.tag}.json", "w"), indent=2)
    json.dump(preds_out, open(RESULTS / f"predictions_{args.model}{args.tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
