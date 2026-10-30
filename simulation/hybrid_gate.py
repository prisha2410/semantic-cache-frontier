"""Does adding a word-overlap check to the embedding cache help?

Idea: TF-IDF notices an added "not" where embeddings do not (baseline.py), so
serve a hit only if BOTH the embedding similarity >= t AND the TF-IDF similarity
between the query and that same cached question >= g. We search every (t, g)
pair for the safe setting, exactly as costs.py does for t alone.

It does not help, and the reason is the finding: near-duplicates share MORE
words with their cached question than genuine paraphrases do (they change one
word; a paraphrase rewords everything), so a word-overlap gate high enough to
stop them also stops the paraphrases.

Output: results/hybrid_gate.csv
    word overlap (median TF-IDF similarity to the embedding's nearest question)
    per query set, and the best safe (t, g) per model and D

Run from the repo root:
    python simulation/hybrid_gate.py
"""
import csv

import numpy as np

from baseline import tfidf_vectors
from costs import BUDGET, CATEGORIES, D_SCENARIOS, R
from simulate import MODELS, OUT, QUERY_SETS, THRESHOLDS, load_index, write_csv

GATES = [round(0.01 * i, 2) for i in range(100)]  # 0.00 ... 0.99


def main():
    index = load_index()
    vectors = tfidf_vectors(index)
    row_of = {r["text_id"]: i for i, r in enumerate(index)}
    out = []
    for model in MODELS:
        with (OUT / f"nearest_{model}.csv").open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        by_set = {}
        for r in rows:
            by_set.setdefault(r["query_set"], []).append((
                float(r["similarity"]),
                float(vectors[row_of[r["query_id"]]] @ vectors[row_of[r["nearest_id"]]]),
                r["answer_correct"] == "True"))
        for name in QUERY_SETS:
            out.append({"model": model, "kind": "median_word_overlap", "query_set": name,
                        "value": round(float(np.median([w for _, w, _ in by_set[name]])), 3)})

        # served[q, t, g] for every query set, as boolean arrays
        t = np.array(THRESHOLDS)[:, None]
        g = np.array(GATES)[None, :]
        def rate(name, wrong_only):
            qs = by_set[name]
            total = np.zeros((len(THRESHOLDS), len(GATES)))
            for sim, word, correct in qs:
                if wrong_only and correct:
                    continue
                total += (sim >= t) & (word >= g)
            return total / len(qs)
        para = rate("paraphrase", False)
        para_err = rate("paraphrase", True)
        mean_err = np.mean([rate(c, True) for c in CATEGORIES], axis=0)
        for d in D_SCENARIOS:
            ok = (R * para_err + d * mean_err) <= BUDGET
            saved = np.where(ok, R * para, -1)
            i, j = np.unravel_index(np.argmax(saved), saved.shape)
            out.append({"model": model, "kind": f"best_safe_d{d}", "query_set": "",
                        "value": f"t={THRESHOLDS[i]} g={GATES[j]} saved={saved[i, j]:.1f}"})
    write_csv(OUT / "hybrid_gate.csv", out)

    print("Median word overlap (TF-IDF) between each query and the embedding's nearest question:")
    for r in out:
        if r["kind"] == "median_word_overlap":
            print(f"  {r['model']:<7}{r['query_set']:<12}{r['value']}")
    print("\nBest safe setting with the word-overlap gate (embedding >= t AND TF-IDF >= g):")
    for r in out:
        if r["kind"].startswith("best"):
            print(f"  {r['model']:<7}{r['kind']:<16}{r['value']}")
    print(f"\nWrote {OUT / 'hybrid_gate.csv'}")


if __name__ == "__main__":
    main()
