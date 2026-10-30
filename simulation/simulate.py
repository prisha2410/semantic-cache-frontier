"""Semantic-cache simulator and threshold sweep.

For every query we find the most similar cached question (cosine similarity =
dot product of normalised embeddings). At threshold t the query is a cache hit
if that similarity >= t. A hit is an ERROR if the cached answer does not match
the query's correct answer; a miss goes to the LLM (assumed correct).

Each query set is tested against its own cache (isolated categories):
    paraphrase, number, entity  -> 787 TriviaQA questions
    comparative, unit, negation -> 787 TriviaQA + that category's constructed
                                   cached questions only

Outputs (results/):
    sweep.csv             model, query_set, threshold, counts, rates, 95% Wilson CIs
    nearest_<model>.csv   per query: nearest cached question, similarity,
                          whether it is the intended target, whether the answer matches

Run from the repo root:
    python simulation/simulate.py
"""
import csv
import json
import math
import re
import string
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
EMB = DATA / "embeddings"
OUT = REPO / "results"

MODELS = ["minilm", "mpnet"]
THRESHOLDS = [round(0.75 + 0.01 * i, 2) for i in range(24)]  # 0.75 ... 0.98
QUERY_SETS = ["paraphrase", "number", "entity", "comparative", "unit", "negation"]


# --------------------------------------------------------------------------- #
# Answer scorer
# --------------------------------------------------------------------------- #

def normalize_answer(text):
    """Lowercase, drop punctuation, articles and extra spaces (TriviaQA-style)."""
    text = text.lower()
    text = "".join(" " if ch in string.punctuation or ch in "‘’“”" else ch for ch in text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def numbers_in(text):
    """All numbers in a string, e.g. '26.2 (26.219)' -> [26.2, 26.219]."""
    out = []
    for m in NUMBER_RE.findall(text):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            pass
    return out


def answers_match(returned, correct_aliases):
    """True if the returned cached answer is a correct answer to the query."""
    ret = normalize_answer(returned)
    if ret and ret in {normalize_answer(a) for a in correct_aliases}:
        return True
    ret_nums = numbers_in(returned)
    if ret_nums:
        for alias in correct_aliases:
            alias_nums = numbers_in(alias)
            if alias_nums and any(math.isclose(r, a, rel_tol=1e-3) for r in ret_nums for a in alias_nums):
                return True
    return False


def wilson(k, n, z=1.96):
    """95% Wilson score interval for k successes out of n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def load_index():
    with (EMB / "embedding_index.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [int(r["row"]) for r in rows] == list(range(len(rows)))
    return rows


def load_aliases():
    """TriviaQA answer aliases per question id (answer + all listed aliases)."""
    questions = json.loads((DATA / "source_questions.json").read_text(encoding="utf-8"))
    return {q["id"]: [q["ground_truth_answer"]] + q["source_metadata"]["aliases"] for q in questions}


def build_sets(index):
    """Row numbers of queries and of the cache for each query set."""
    base_cache = [i for i, r in enumerate(index) if r["role"] == "cache" and r["category"] == "triviaqa"]
    row_of = {r["text_id"]: i for i, r in enumerate(index)}
    sets = {}
    for name in QUERY_SETS:
        if name == "paraphrase":
            queries = [i for i, r in enumerate(index) if r["role"] == "query_paraphrase"]
            cache = base_cache
        else:
            queries = [i for i, r in enumerate(index)
                       if r["role"] == "query_adversarial" and r["category"] == name]
            own = sorted({row_of[index[i]["target_cache_id"]] for i in queries
                          if index[row_of[index[i]["target_cache_id"]]]["category"] == "constructed"})
            cache = base_cache + own
        sets[name] = (queries, cache)
    return sets


def correct_aliases(row, aliases):
    """All acceptable answers for a query."""
    if row["role"] == "query_paraphrase":
        return aliases[row["target_cache_id"]]
    return [row["answer"]]


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #

def simulate_set(vectors, index, aliases, queries, cache):
    """Nearest cached question for each query, plus answer correctness."""
    sims = vectors[queries] @ vectors[cache].T
    best = sims.argmax(axis=1)
    out = []
    for qpos, qrow in enumerate(queries):
        crow = cache[best[qpos]]
        q, c = index[qrow], index[crow]
        out.append({
            "query_id": q["text_id"], "query": q["text"], "correct_answer": q["answer"],
            "nearest_id": c["text_id"], "nearest_question": c["text"], "cached_answer": c["answer"],
            "similarity": float(sims[qpos, best[qpos]]),
            "is_target": c["text_id"] == q["target_cache_id"],
            "answer_correct": answers_match(c["answer"], correct_aliases(q, aliases)),
        })
    return out


def sweep(per_query):
    """Counts and rates at every threshold."""
    rows = []
    n = len(per_query)
    for t in THRESHOLDS:
        hits = [r for r in per_query if r["similarity"] >= t]
        correct = sum(r["answer_correct"] for r in hits)
        errors = len(hits) - correct
        hit_lo, hit_hi = wilson(len(hits), n)
        err_lo, err_hi = wilson(errors, n)
        rows.append({"threshold": t, "n": n, "hits": len(hits), "correct_hits": correct, "errors": errors,
                     "hit_rate": len(hits) / n, "hit_rate_lo": hit_lo, "hit_rate_hi": hit_hi,
                     "error_rate": errors / n, "error_rate_lo": err_lo, "error_rate_hi": err_hi,
                     "hit_precision": correct / len(hits) if hits else float("nan")})
    return rows


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(exist_ok=True)
    index = load_index()
    aliases = load_aliases()
    sets = build_sets(index)
    all_sweeps = []
    for model in MODELS:
        vectors = np.load(EMB / f"{model}.npy")
        assert vectors.shape[0] == len(index)
        nearest = []
        for name, (queries, cache) in sets.items():
            per_query = simulate_set(vectors, index, aliases, queries, cache)
            for r in per_query:
                r.update({"model": model, "query_set": name, "cache_size": len(cache)})
            nearest += per_query
            for r in sweep(per_query):
                all_sweeps.append({"model": model, "query_set": name, "cache_size": len(cache), **r})
            top_target = np.mean([r["is_target"] for r in per_query])
            print(f"{model:6s} {name:11s} n={len(queries):4d} cache={len(cache)} "
                  f"nearest-is-target={top_target:.3f}")
        cols = ["model", "query_set", "cache_size", "query_id", "query", "correct_answer", "nearest_id",
                "nearest_question", "cached_answer", "similarity", "is_target", "answer_correct"]
        write_csv(OUT / f"nearest_{model}.csv", [{k: r[k] for k in cols} for r in nearest])
    write_csv(OUT / "sweep.csv", all_sweeps)
    print(f"\nWrote {OUT / 'sweep.csv'} ({len(all_sweeps)} rows) and nearest_<model>.csv")


if __name__ == "__main__":
    main()
