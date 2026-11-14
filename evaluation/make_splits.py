"""Context-level (grouped) train/validation/test split of HEA_QA_DATASET.json.

All questions generated from the same context (alloy record) go to the same split, so
paraphrased questions about one fact can never appear on both sides. The script also
quantifies the leakage a naive random question-level split would have caused.
"""
import json
import random

import numpy as np

from common import RESULTS, SEED, SPLITS, load_squad, normalize_answer

FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}


def grouped_split(contexts, seed=SEED):
    ids = [c["cid"] for c in contexts]
    rng = random.Random(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_test = round(FRACTIONS["test"] * n)
    n_val = round(FRACTIONS["val"] * n)
    return {"test": sorted(ids[:n_test]), "val": sorted(ids[n_test:n_test + n_val]),
            "train": sorted(ids[n_test + n_val:])}


def leakage_of_random_split(contexts, seed=SEED):
    """Share of test questions whose (context, answer) fact is also in train under a random QA split."""
    rows = [(c["cid"], q) for c in contexts for q in c["qas"]]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(rows))
    n_test = round(0.15 * len(rows))
    test, train = [rows[i] for i in perm[:n_test]], [rows[i] for i in perm[n_test:]]
    train_ctx = {cid for cid, _ in train}
    train_fact = {(cid, normalize_answer(q["answer"])) for cid, q in train}
    same_ctx = np.mean([cid in train_ctx for cid, _ in test])
    same_fact = np.mean([(cid, normalize_answer(q["answer"])) in train_fact for cid, q in test])
    return {"random_split_test_questions": n_test,
            "test_questions_with_context_seen_in_train": float(same_ctx),
            "test_questions_with_same_fact_seen_in_train": float(same_fact)}


def main():
    SPLITS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    contexts = load_squad()
    split = grouped_split(contexts)
    by_id = {c["cid"]: c for c in contexts}
    summary = {}
    for name, ids in split.items():
        recs = [by_id[i] for i in ids]
        json.dump(recs, open(SPLITS / f"{name}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        summary[name] = {"contexts": len(recs), "questions": sum(len(c["qas"]) for c in recs)}
    test_ctx = {normalize_answer(by_id[i]["context"]) for i in split["test"]}
    train_ctx = {normalize_answer(by_id[i]["context"]) for i in split["train"]}
    summary["overlap_identical_contexts_train_test"] = len(test_ctx & train_ctx)
    summary["leakage_if_random_question_split"] = leakage_of_random_split(contexts)
    json.dump(summary, open(RESULTS / "split_summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
