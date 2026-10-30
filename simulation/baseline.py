"""Baselines: an exact-match cache and a word-overlap (TF-IDF) cache.

Both use the same query sets, the same isolated caches and the same answer
scorer as simulate.py, so they compare directly with the embedding models.

exact   a hit only if the query text equals a cached question's text after
        lowercasing, removing punctuation and collapsing spaces (a normal,
        non-semantic cache).
tfidf   cosine similarity of TF-IDF word vectors (scikit-learn defaults), with the
        vocabulary and IDF weights learned from all cached questions. Its
        similarities are lower than the embedding models', so it is swept over a
        wider range of thresholds (0.30-0.98).

Outputs (results/):
    baseline_exact.csv   one row per query set: hits, errors, rates, 95% Wilson CIs
    sweep_tfidf.csv      same columns as sweep.csv, model = tfidf
    nearest_tfidf.csv    same columns as nearest_<model>.csv

Run from the repo root:
    python simulation/baseline.py
"""
import string

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from simulate import (OUT, answers_match, build_sets, correct_aliases, load_aliases, load_index,
                      simulate_set, sweep, wilson, write_csv)

TFIDF_THRESHOLDS = [round(0.30 + 0.01 * i, 2) for i in range(69)]  # 0.30 ... 0.98


def normalize_question(text):
    """Lowercase, drop punctuation and extra spaces."""
    text = text.lower()
    text = "".join(" " if ch in string.punctuation or ch in "‘’“”" else ch for ch in text)
    return " ".join(text.split())


def exact_match(index, aliases, sets):
    rows = []
    for name, (queries, cache) in sets.items():
        lookup = {normalize_question(index[c]["text"]): index[c] for c in cache}
        hits = errors = 0
        for qrow in queries:
            q = index[qrow]
            c = lookup.get(normalize_question(q["text"]))
            if c is not None:
                hits += 1
                errors += not answers_match(c["answer"], correct_aliases(q, aliases))
        n = len(queries)
        hit_lo, hit_hi = wilson(hits, n)
        err_lo, err_hi = wilson(errors, n)
        rows.append({"model": "exact", "query_set": name, "n": n, "hits": hits, "errors": errors,
                     "hit_rate": hits / n, "hit_rate_lo": hit_lo, "hit_rate_hi": hit_hi,
                     "error_rate": errors / n, "error_rate_lo": err_lo, "error_rate_hi": err_hi})
    return rows


def tfidf_vectors(index):
    """L2-normalised TF-IDF vectors, so cosine similarity = dot product (as for the embeddings)."""
    texts = [r["text"] for r in index]
    vectorizer = TfidfVectorizer().fit([r["text"] for r in index if r["role"] == "cache"])
    return vectorizer.transform(texts).toarray().astype(np.float32)


def main():
    index = load_index()
    aliases = load_aliases()
    sets = build_sets(index)

    exact = exact_match(index, aliases, sets)
    write_csv(OUT / "baseline_exact.csv", exact)
    print("Exact-match cache:")
    for r in exact:
        print(f"  {r['query_set']:<12} n={r['n']:<5} hits {r['hits']:<4} errors {r['errors']}")

    vectors = tfidf_vectors(index)
    nearest, sweeps = [], []
    print("\nTF-IDF cache:")
    for name, (queries, cache) in sets.items():
        per_query = simulate_set(vectors, index, aliases, queries, cache)
        for r in per_query:
            r.update({"model": "tfidf", "query_set": name, "cache_size": len(cache)})
        nearest += per_query
        for r in sweep(per_query, TFIDF_THRESHOLDS):
            sweeps.append({"model": "tfidf", "query_set": name, "cache_size": len(cache), **r})
        top_target = np.mean([r["is_target"] for r in per_query])
        print(f"  {name:<12} n={len(queries):<5} nearest-is-target={top_target:.3f}")
    cols = ["model", "query_set", "cache_size", "query_id", "query", "correct_answer", "nearest_id",
            "nearest_question", "cached_answer", "similarity", "is_target", "answer_correct"]
    write_csv(OUT / "nearest_tfidf.csv", [{k: r[k] for k in cols} for r in nearest])
    write_csv(OUT / "sweep_tfidf.csv", sweeps)
    print(f"\nWrote {OUT / 'baseline_exact.csv'}, {OUT / 'sweep_tfidf.csv'}, {OUT / 'nearest_tfidf.csv'}")


if __name__ == "__main__":
    main()
