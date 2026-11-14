"""Retrieval evaluation of the RAG component (all-MiniLM-L6-v2 + FAISS, cosine similarity).

Knowledge base = every context of HEA_QA_DATASET.json. Queries = held-out test questions
(and validation questions). A hit means the gold context of the question is retrieved.
"""
import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from common import RESULTS, bootstrap_ci, load_split, load_squad

KS = (1, 3, 5)


def dense_scores(model, contexts, queries):
    """Cosine similarity via FAISS inner product on L2-normalised MiniLM embeddings."""
    emb = model.encode([c["context"] for c in contexts], normalize_embeddings=True, convert_to_numpy=True)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb.astype(np.float32))
    q_emb = model.encode([q for q, _ in queries], normalize_embeddings=True, convert_to_numpy=True)
    sims, idx = index.search(q_emb.astype(np.float32), len(contexts))
    full = np.zeros_like(sims)
    np.put_along_axis(full, idx, sims, axis=1)
    return full


def lexical_scores(contexts, queries):
    """Character n-gram TF-IDF cosine; matches alloy formulae such as MoNbVTa0.5Cr exactly."""
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
    c = vec.fit_transform([x["context"] for x in contexts])
    q = vec.transform([x for x, _ in queries])
    return (q @ c.T).toarray()


def rank_by(scores):
    idx = np.argsort(-scores, axis=1)
    return np.take_along_axis(scores, idx, axis=1), idx


def evaluate(scores, contexts, queries):
    ctx_ids = [c["cid"] for c in contexts]
    sims, idx = rank_by(scores)
    ranks, gold_sim, best_wrong = [], [], []
    for row, srow, (_, gold) in zip(idx, sims, queries):
        retrieved = [ctx_ids[i] for i in row]
        r = retrieved.index(gold) + 1
        ranks.append(r)
        gold_sim.append(float(srow[r - 1]))
        best_wrong.append(float(next(s for cid, s in zip(retrieved, srow) if cid != gold)))
    ranks = np.array(ranks)
    groups = [g for _, g in queries]
    out = {"queries": len(queries), "contexts_in_index": len(contexts)}
    for k in KS:
        hits = (ranks <= k).astype(float)
        out[f"recall@{k}"] = float(hits.mean())
        out[f"recall@{k}_95ci"] = bootstrap_ci(hits, groups)
    rr = 1.0 / ranks
    out["mrr"] = float(rr.mean())
    out["mrr_95ci"] = bootstrap_ci(rr, groups)
    out["mean_cosine_gold"] = float(np.mean(gold_sim))
    out["mean_cosine_best_non_gold"] = float(np.mean(best_wrong))
    return out, idx


def zscore(x):
    return (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-9)


def main():
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    contexts = load_squad()
    ctx_ids = [c["cid"] for c in contexts]
    results, retrieved = {}, {}
    for name in ("val", "test"):
        split = load_split(name)
        queries = [(q["question"], c["cid"]) for c in split for q in c["qas"]]
        dense = dense_scores(model, contexts, queries)
        lex = lexical_scores(contexts, queries)
        hybrid = 0.5 * zscore(dense) + 0.5 * zscore(lex)
        results[name] = {}
        for label, sc in (("dense_minilm", dense), ("lexical_tfidf_char", lex), ("hybrid", hybrid)):
            results[name][label], idx = evaluate(sc, contexts, queries)
            if label == "hybrid":
                retrieved[name] = {q["id"]: [ctx_ids[i] for i in row[:5]]
                                   for q, row in zip((q for c in split for q in c["qas"]), idx)}
    json.dump(results, open(RESULTS / "retrieval_metrics.json", "w"), indent=2)
    json.dump(retrieved, open(RESULTS / "retrieved_contexts_hybrid.json", "w"))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
