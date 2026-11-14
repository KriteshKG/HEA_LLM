"""Automatic validation of the generated Q&A pairs (HEA_QA_DATASET.json).

Checks: answer grounded verbatim in its context, answer_start offset consistent,
exact/near duplicate questions, answer type, and question category. Also writes a
stratified random sample for manual expert audit.
"""
import csv
import json
import random
import re
from collections import Counter

from common import RESULTS, SEED, load_squad, normalize_answer

CATEGORY_RULES = [
    ("Unit-based", re.compile(r"\bunits?\b", re.I)),
    ("Temperature-based", re.compile(r"\b(at what temperature|which temperature|what temperature|temperature (?:is|was|were|at which))", re.I)),
    ("Material-condition-based", re.compile(r"\b(phase|equilibrium|single|multiphase|microstructure|structure|condition)", re.I)),
    ("Property-value-based", re.compile(r".*")),
]
NUM = re.compile(r"\d")


def category(q):
    return next(name for name, rx in CATEGORY_RULES if rx.search(q))


def main():
    contexts = load_squad()
    rows = []
    for c in contexts:
        ctx_norm = normalize_answer(c["context"])
        for q in c["qas"]:
            a = q["answer"]
            rows.append({
                "cid": c["cid"], "id": q["id"], "question": q["question"], "answer": a,
                "category": category(q["question"]),
                "offset_ok": c["context"][q["answer_start"]:q["answer_start"] + len(a)] == a,
                "verbatim_in_context": a in c["context"],
                "normalised_in_context": normalize_answer(a) in ctx_norm,
                "numeric_answer": bool(NUM.search(a)),
                "answer_words": len(a.split()),
            })
    n = len(rows)
    q_norm = Counter(normalize_answer(r["question"]) for r in rows)
    qa_norm = Counter((normalize_answer(r["question"]), normalize_answer(r["answer"])) for r in rows)
    facts = Counter((r["cid"], normalize_answer(r["answer"])) for r in rows)
    conflicting = {}
    for r in rows:
        conflicting.setdefault((r["cid"], normalize_answer(r["question"])), set()).add(normalize_answer(r["answer"]))
    summary = {
        "contexts": len(contexts),
        "qa_pairs": n,
        "answer_offset_consistent": sum(r["offset_ok"] for r in rows) / n,
        "answer_verbatim_in_context": sum(r["verbatim_in_context"] for r in rows) / n,
        "answer_normalised_in_context": sum(r["normalised_in_context"] for r in rows) / n,
        "numeric_answers": sum(r["numeric_answer"] for r in rows) / n,
        "median_answer_words": sorted(r["answer_words"] for r in rows)[n // 2],
        "duplicate_questions_exact_after_normalisation": n - len(q_norm),
        "duplicate_question_answer_pairs": n - len(qa_norm),
        "same_question_conflicting_answers": sum(1 for v in conflicting.values() if len(v) > 1),
        "distinct_facts_(context,answer)": len(facts),
        "mean_paraphrases_per_fact": n / len(facts),
        "questions_per_context_min_median_max": [
            min(len(c["qas"]) for c in contexts),
            sorted(len(c["qas"]) for c in contexts)[len(contexts) // 2],
            max(len(c["qas"]) for c in contexts)],
        "category_counts": dict(Counter(r["category"] for r in rows)),
    }
    json.dump(summary, open(RESULTS / "qa_validation_summary.json", "w"), indent=2)
    with open(RESULTS / "qa_ungrounded_pairs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(r for r in rows if not r["verbatim_in_context"] or not r["offset_ok"])
    rng = random.Random(SEED)
    audit = []
    for cat in sorted({r["category"] for r in rows}):
        pool = [r for r in rows if r["category"] == cat]
        audit += rng.sample(pool, min(25, len(pool)))
    ctx = {c["cid"]: c["context"] for c in contexts}
    with open(RESULTS / "qa_manual_audit_sheet.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "category", "context", "question", "answer",
                    "relevant(0/1)", "factually_consistent(0/1)", "grounded_in_context(0/1)", "rater_notes"])
        for r in audit:
            w.writerow([r["id"], r["category"], ctx[r["cid"]], r["question"], r["answer"], "", "", "", ""])
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
