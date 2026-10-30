"""Real-traffic check: how does the cache behave on real TriviaQA questions?

Our main results use queries WE made (paraphrases and edited near-duplicates)
and an assumed traffic mix. Here the traffic is real: 2,000 TriviaQA questions
drawn at random from the same filtered pool (62,625) that our 800 came from,
sent to a cache holding the rest of the pool (60,625 questions). Nothing is
edited or written by us, so this measures how often real, distinct questions
land close to a cached question with a different answer.

Two steps:

    python simulation/real_traffic.py build     # GPU, once (~10 min, our estimate)
        loads TriviaQA, rebuilds the filtered pool with data/build_source_questions.py,
        embeds it with both models, finds each traffic question's nearest cached
        question, runs the rule guard and the LLM judge on every pair with
        similarity >= 0.50, and writes
            results/real_traffic_pairs.csv     one row per (model, traffic question)
            results/real_traffic_judge.jsonl   judge verdicts (reused on a rerun)

    python simulation/real_traffic.py analyse   # CPU, seconds; part of reproduce.py
        per model, verifier and threshold: hits and wrong answers per 1,000 real
        questions, plus the settings costs.py recommends, and writes
            results/real_traffic.csv
            results/real_traffic_wrong_examples.csv   every wrong answer served at the
                                                      recommended settings, to read by hand

A "wrong answer" uses the same answer scorer as simulate.py: the cached
question's answer is checked against the traffic question's answer and aliases.
A hit on a DIFFERENT question with the SAME answer counts as correct, as it would
for a user.
"""
import csv
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

from costs import VERIFIERS
from rule_guard import rule_guard
from simulate import MODELS, OUT, THRESHOLDS, answers_match, wilson, write_csv

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "data"))

PAIRS = OUT / "real_traffic_pairs.csv"
JUDGE_CACHE = OUT / "real_traffic_judge.jsonl"
SUMMARY = OUT / "real_traffic.csv"
EXAMPLES = OUT / "real_traffic_wrong_examples.csv"

N_TRAFFIC = 2000
SEED = 7
MIN_JUDGE = 0.50
EMBED_MODELS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
}
# The settings costs.py recommends (results/safe_thresholds.csv, D = 10)
RECOMMENDED = [("minilm", "rule", 0.61), ("mpnet", "rule", 0.64),
               ("minilm", "both", 0.50), ("mpnet", "both", 0.50)]


def clean(question):
    """Undo CSV-style quoting, as for our cached questions (generate_paraphrases.py)."""
    if question.startswith('"') and question.endswith('"') and '""' in question:
        question = question[1:-1].replace('""', '"')
    return question


def load_pool(limit=None):
    """The filtered, de-duplicated TriviaQA pool our 800 questions were sampled from."""
    from build_source_questions import FilterConfig, _norm_question, evaluate, load_triviaqa

    cfg, seen, pool = FilterConfig(), set(), []
    for rec in load_triviaqa():
        if evaluate(rec, cfg)[0]:
            continue
        key = _norm_question(rec["question"])
        if key in seen:
            continue
        seen.add(key)
        ans = rec["answer"]
        pool.append({"qid": rec["question_id"], "question": clean(rec["question"].strip()),
                     "answer": (ans.get("value") or "").strip(),
                     "aliases": [(ans.get("value") or "").strip()] + list(ans.get("aliases") or [])})
        if limit and len(pool) >= limit:
            break
    return pool


def build(limit=None):
    import torch
    from sentence_transformers import SentenceTransformer

    from verify import make_qwen_judge, parse_verdict

    pool = load_pool(limit)
    rng = random.Random(SEED)
    n_traffic = min(N_TRAFFIC, len(pool) // 10) if limit else N_TRAFFIC
    traffic_idx = sorted(rng.sample(range(len(pool)), n_traffic))
    is_traffic = np.zeros(len(pool), dtype=bool)
    is_traffic[traffic_idx] = True
    cache_idx = np.flatnonzero(~is_traffic)
    print(f"Pool {len(pool)} questions: traffic {len(traffic_idx)}, cache {len(cache_idx)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    texts = [p["question"] for p in pool]
    rows = []
    for short, name in EMBED_MODELS.items():
        print(f"\nEncoding {len(texts)} questions with {name} on {device} ...")
        start = time.perf_counter()
        model = SentenceTransformer(name, device=device)
        vectors = model.encode(texts, batch_size=128, convert_to_numpy=True,
                               normalize_embeddings=True, show_progress_bar=True).astype(np.float32)
        print(f"  {time.perf_counter() - start:.0f}s")
        cache_vectors = vectors[cache_idx]
        for start_row in range(0, len(traffic_idx), 500):
            chunk = traffic_idx[start_row:start_row + 500]
            sims = vectors[chunk] @ cache_vectors.T
            best = sims.argmax(axis=1)
            for k, qi in enumerate(chunk):
                q, c = pool[qi], pool[cache_idx[best[k]]]
                accept, reason = rule_guard(q["question"], c["question"])
                rows.append({"model": short, "query_id": q["qid"], "query": q["question"],
                             "query_answer": q["answer"], "nearest_id": c["qid"],
                             "nearest_question": c["question"], "cached_answer": c["answer"],
                             "similarity": float(sims[k, best[k]]),
                             "answer_correct": answers_match(c["answer"], q["aliases"]),
                             "rule_accept": accept, "rule_reason": reason})
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    # LLM judge on every pair that could be a hit, each unique pair once, resumable.
    cache = {}
    if JUDGE_CACHE.exists():
        for line in JUDGE_CACHE.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            cache[(item["query"], item["cached"])] = item["raw"]
    todo = list(dict.fromkeys((r["query"], r["nearest_question"]) for r in rows
                              if r["similarity"] >= MIN_JUDGE and (r["query"], r["nearest_question"]) not in cache))
    print(f"\nJudging {len(todo)} pairs ({len(cache)} already judged)")
    if todo:
        judge, _, _ = make_qwen_judge()
        start = time.time()
        with JUDGE_CACHE.open("a", encoding="utf-8") as f:
            for i, (query, cached) in enumerate(todo, 1):
                raw = judge(query, cached)
                cache[(query, cached)] = raw
                f.write(json.dumps({"query": query, "cached": cached, "raw": raw}, ensure_ascii=False) + "\n")
                f.flush()
                if i % 200 == 0 or i == len(todo):
                    elapsed = time.time() - start
                    print(f"  {i}/{len(todo)}  {elapsed:.0f}s elapsed, ~{elapsed / i * (len(todo) - i):.0f}s left")
    for r in rows:
        raw = cache.get((r["query"], r["nearest_question"]), "")
        r["judge_raw"] = raw
        r["judge_accept"] = parse_verdict(raw) is True
    write_csv(PAIRS, rows)
    print(f"\nWrote {PAIRS} ({len(rows)} rows) and {JUDGE_CACHE}")


def accepts(verifier, r):
    rule, judge = r["rule_accept"] == "True", r["judge_accept"] == "True"
    return {"none": True, "rule": rule, "judge": judge, "both": rule and judge}[verifier]


def analyse():
    if not PAIRS.exists():
        print(f"{PAIRS.name} not found: run `python simulation/real_traffic.py build` first (GPU)")
        return
    with PAIRS.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for model in MODELS:
        pairs = [r for r in rows if r["model"] == model]
        n = len(pairs)
        for verifier in VERIFIERS:
            for t in THRESHOLDS:
                served = [r for r in pairs if float(r["similarity"]) >= t and accepts(verifier, r)]
                wrong = sum(r["answer_correct"] == "False" for r in served)
                lo, hi = wilson(wrong, n)
                out.append({"model": model, "verifier": verifier, "threshold": t, "n": n,
                            "served": len(served), "correct_served": len(served) - wrong, "wrong": wrong,
                            "served_per_1000": round(1000 * len(served) / n, 1),
                            "wrong_per_1000": round(1000 * wrong / n, 1),
                            "wrong_per_1000_lo": round(1000 * lo, 1), "wrong_per_1000_hi": round(1000 * hi, 1)})
    write_csv(SUMMARY, out)

    examples = []
    print(f"Real traffic: {len(rows) // len(MODELS)} TriviaQA questions sent to a cache of the rest of the pool")
    print(f"  {'model':<7}{'verifier':<9}{'t':>5}  {'served per 1,000':>17}  {'wrong per 1,000 (95% CI)':>26}")
    for model, verifier, t in [(m, "none", t) for m in MODELS for t in (0.75, 0.90)] + RECOMMENDED:
        r = next(x for x in out if (x["model"], x["verifier"], x["threshold"]) == (model, verifier, t))
        print(f"  {model:<7}{verifier:<9}{t:>5}  {r['served_per_1000']:>17}  "
              f"{r['wrong_per_1000']:>8} ({r['wrong_per_1000_lo']}–{r['wrong_per_1000_hi']})")
        if (model, verifier, t) in RECOMMENDED:
            for p in rows:
                if (p["model"] == model and float(p["similarity"]) >= t and accepts(verifier, p)
                        and p["answer_correct"] == "False"):
                    examples.append({"setting": f"{model} {verifier} t={t}", **p})
    if examples:
        write_csv(EXAMPLES, examples)
    else:
        EXAMPLES.write_text("no wrong answers served\n", encoding="utf-8")
    print(f"\nWrote {SUMMARY} and {EXAMPLES} ({len(examples)} wrong answers at the recommended settings)")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "analyse"
    if mode == "build":
        limit = int(sys.argv[2]) if len(sys.argv) > 2 else None  # test runs only
        build(limit)
    else:
        analyse()
