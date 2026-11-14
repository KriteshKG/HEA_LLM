"""Shared helpers for the HEA_QA evaluation scripts."""
import json
import re
import string
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Datasets"
SPLITS = ROOT / "evaluation" / "splits"
RESULTS = ROOT / "evaluation" / "results"
SEED = 42


def load_squad(path=DATA / "HEA_QA_DATASET.json"):
    """Return a list of contexts, each {'cid', 'context', 'qas': [{'id','question','answer','answer_start'}]}."""
    raw = json.load(open(path, encoding="utf-8"))["data"]
    contexts = []
    for item in raw.values():
        for pid, para in item["paragraphs"].items():
            qas = []
            for qa in para["qas"].values():
                a = qa["answers"]["0"]
                qas.append({"id": qa["id"], "question": qa["question"],
                            "answer": a["text"], "answer_start": a["answer_start"]})
            contexts.append({"cid": int(pid), "context": para["context"], "qas": qas})
    return contexts


def normalize_answer(s):
    """SQuAD v1.1 answer normalisation."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred, gold):
    return float(normalize_answer(pred) == normalize_answer(gold))


def f1_score(pred, gold):
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if not p or not g:
        return float(p == g)
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


def wilson_ci(k, n, z=1.96):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_ci(values, groups=None, n_boot=2000, seed=SEED):
    """Percentile 95% CI of the mean; resamples whole groups (contexts) when given."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    if groups is None:
        idx = [rng.integers(0, len(values), len(values)) for _ in range(n_boot)]
        means = [values[i].mean() for i in idx]
    else:
        groups = np.asarray(groups)
        uniq = np.unique(groups)
        members = {g: np.where(groups == g)[0] for g in uniq}
        means = []
        for _ in range(n_boot):
            pick = rng.choice(uniq, len(uniq), replace=True)
            means.append(values[np.concatenate([members[g] for g in pick])].mean())
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def load_split(name):
    return json.load(open(SPLITS / f"{name}.json", encoding="utf-8"))
