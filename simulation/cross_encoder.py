"""A third verifier: a cross-encoder trained to spot duplicate questions.

The rule guard only sees the edits we designed, and the 1.5B LLM judge waves
through many "same topic, different question" pairs on real traffic. A
cross-encoder reads both questions together and outputs one score: how likely
they are the same question. cross-encoder/quora-distilroberta-base was trained
on Quora duplicate-question pairs, which is close to the cache's question.

The cut-off c is CHOSEN on the synthetic data and TESTED on real traffic, so the
real-traffic numbers are a fair out-of-sample test:
    1. synthetic: for each c, the safe threshold and calls saved at D = 10
       (same cost model as costs.py); c* = the c that saves the most
    2. real traffic: with c* (and with the model's natural cut-off 0.5),
       label-corrected wrong and correct answers per 1,000 real questions,
       and the real-traffic safe threshold, as in real_labels.py
    3. an upper bound: the best (c, t) chosen on the real traffic itself, for a
       budget of 1 and of 5 wrong answers per 1,000. This is NOT a fair test (it
       is tuned on the data it is scored on); it shows the most any cut-off could do.

    python simulation/cross_encoder.py score     # GPU (or a slow CPU), once
        scores every judged pair from the synthetic test and from real traffic,
        writes results/cross_scores.jsonl
    python simulation/cross_encoder.py analyse   # CPU, seconds; part of reproduce.py
        writes results/cross_encoder.csv
"""
import csv
import json
import math
import sys

from costs import BUDGET, CATEGORIES, D_SCENARIOS, R, load
from real_labels import CHECK as LABEL_CHECK, group
from real_traffic import PAIRS
from simulate import MODELS, OUT, THRESHOLDS, write_csv

MODEL_NAME = "cross-encoder/quora-distilroberta-base"
SCORES = OUT / "cross_scores.jsonl"
SUMMARY = OUT / "cross_encoder.csv"
MIN_SIMILARITY = 0.50
CUTOFFS = [round(0.05 * i, 2) for i in range(1, 20)]  # 0.05 ... 0.95
UPPER_BUDGETS = [1.0, 5.0]  # wrong answers per 1,000, for the upper bound
VERIFIERS = {
    "cross": lambda v, ok: ok,
    "rule+cross": lambda v, ok: v["rule_accept"] == "True" and ok,
}


def pairs_to_score():
    pairs = set()
    for model in MODELS:
        with (OUT / f"verdicts_{model}.csv").open(encoding="utf-8", newline="") as f:
            pairs |= {(r["query"], r["nearest_question"]) for r in csv.DictReader(f)}
    with PAIRS.open(encoding="utf-8", newline="") as f:
        pairs |= {(r["query"], r["nearest_question"]) for r in csv.DictReader(f)
                  if float(r["similarity"]) >= MIN_SIMILARITY}
    return sorted(pairs)


def score():
    import torch
    from sentence_transformers import CrossEncoder

    pairs = pairs_to_score()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Scoring {len(pairs)} pairs with {MODEL_NAME} on {device} ...")
    model = CrossEncoder(MODEL_NAME, device=device)
    raw = model.predict([list(p) for p in pairs], batch_size=64, show_progress_bar=True)
    with SCORES.open("w", encoding="utf-8") as f:
        for (query, cached), s in zip(pairs, raw):
            s = float(s)
            # The model should already return a probability; if it returns a logit, squash it.
            prob = s if 0.0 <= s <= 1.0 else 1 / (1 + math.exp(-s))
            f.write(json.dumps({"query": query, "cached": cached, "score": prob}, ensure_ascii=False) + "\n")
    print(f"Wrote {SCORES}")


def load_scores():
    scores = {}
    for line in SCORES.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        scores[(item["query"], item["cached"])] = item["score"]
    return scores


def synthetic_safe(by_set, scores, verifier, c, d):
    """Best threshold on the synthetic test for one verifier, cut-off and D (point estimate)."""
    def served(q, t):
        v = q["verdict"]
        if q["similarity"] < t or v is None:
            return False
        return VERIFIERS[verifier](v, scores[(v["query"], v["nearest_question"])] >= c)
    best = None
    for t in THRESHOLDS:
        para = by_set["paraphrase"]
        saved = R * sum(served(q, t) for q in para) / len(para)
        wrong = R * sum(served(q, t) and not q["answer_correct"] for q in para) / len(para)
        wrong += d * sum(sum(served(q, t) and not q["answer_correct"] for q in by_set[cat]) / len(by_set[cat])
                         for cat in CATEGORIES) / len(CATEGORIES)
        if wrong <= BUDGET and (best is None or saved > best[1]):
            best = (t, saved, wrong)
    return best


def upper_bound(rows, scores, wrong_rate, verifier, budget):
    """Best (c, t) on real traffic itself: the most correct answers within the budget."""
    best = None
    for c in CUTOFFS:
        for t, correct, wrong in real_traffic(rows, scores, wrong_rate, verifier, c):
            if wrong <= budget and (best is None or correct > best[2]):
                best = (c, t, correct, wrong)
    return best


def real_traffic(rows, scores, wrong_rate, verifier, c):
    """Per threshold: label-corrected wrong and correct answers per 1,000 real questions."""
    out = []
    for t in THRESHOLDS:
        served = [r for r in rows if float(r["similarity"]) >= t
                  and VERIFIERS[verifier](r, scores.get((r["query"], r["nearest_question"]), 0.0) >= c)]
        wrong = 1000 * sum(wrong_rate[group(r)] for r in served) / len(rows)
        out.append((t, 1000 * len(served) / len(rows) - wrong, wrong))
    return out


def analyse():
    if not SCORES.exists():
        print(f"{SCORES.name} not found: run `python simulation/cross_encoder.py score` first")
        return
    scores = load_scores()
    labels = json.loads(LABEL_CHECK.read_text(encoding="utf-8"))
    wrong_rate = {g: v["rate"] for g, v in labels["wrong_rate_by_group"].items()}
    with PAIRS.open(encoding="utf-8", newline="") as f:
        all_rows = list(csv.DictReader(f))

    out = []
    for model in MODELS:
        by_set = load(model)
        rows = [r for r in all_rows if r["model"] == model]
        for verifier in VERIFIERS:
            # 1. choose c on the synthetic data (D = 10); among equal savings, the stricter c
            tried = {c: synthetic_safe(by_set, scores, verifier, c, D_SCENARIOS[0]) for c in CUTOFFS}
            c_star = max((c for c in CUTOFFS if tried[c]), key=lambda c: (round(tried[c][1], 6), c), default=None)
            for label, c in (("chosen on synthetic", c_star), ("natural 0.5", 0.5)):
                if c is None:
                    continue
                row = {"model": model, "verifier": verifier, "cutoff": c, "cutoff_source": label}
                for d in D_SCENARIOS:
                    s = synthetic_safe(by_set, scores, verifier, c, d)
                    row[f"synthetic_d{d}_threshold"] = s[0] if s else "none"
                    row[f"synthetic_d{d}_saved"] = round(s[1], 1) if s else 0.0
                # 2. test on real traffic
                curve = real_traffic(rows, scores, wrong_rate, verifier, c)
                t_syn = row["synthetic_d10_threshold"]
                at = next((x for x in curve if x[0] == t_syn), None)
                row["real_at_synthetic_t_correct"] = round(at[1], 1) if at else ""
                row["real_at_synthetic_t_wrong"] = round(at[2], 2) if at else ""
                safe = max((x for x in curve if x[2] <= BUDGET), key=lambda x: x[1], default=None)
                row["real_safe_threshold"] = safe[0] if safe else "none"
                row["real_safe_correct"] = round(safe[1], 1) if safe else 0.0
                row["real_safe_wrong"] = round(safe[2], 2) if safe else ""
                out.append(row)
    upper = []
    for model in MODELS:
        rows = [r for r in all_rows if r["model"] == model]
        for verifier in VERIFIERS:
            for budget in UPPER_BUDGETS:
                b = upper_bound(rows, scores, wrong_rate, verifier, budget)
                upper.append({"model": model, "verifier": verifier, "cutoff": b[0] if b else "none",
                              "cutoff_source": f"upper bound, budget {budget:g}",
                              "real_safe_threshold": b[1] if b else "none",
                              "real_safe_correct": round(b[2], 1) if b else 0.0,
                              "real_safe_wrong": round(b[3], 2) if b else ""})
    write_csv(SUMMARY, out + upper)

    print(f"Cross-encoder verifier ({MODEL_NAME}); cut-off c chosen on synthetic data, tested on real traffic")
    print(f"  {'model':<7}{'verifier':<11}{'c':>5} {'source':<20}{'synthetic D=10':<20}"
          f"{'real at that t':<24}{'real safe t'}")
    for r in out:
        syn = f"t={r['synthetic_d10_threshold']} saved {r['synthetic_d10_saved']}"
        real = (f"{r['real_at_synthetic_t_wrong']} wrong / {r['real_at_synthetic_t_correct']} ok"
                if r["real_at_synthetic_t_wrong"] != "" else "–")
        safe = (f"t={r['real_safe_threshold']}, {r['real_safe_correct']} ok, {r['real_safe_wrong']} wrong"
                if r["real_safe_threshold"] != "none" else "none")
        print(f"  {r['model']:<7}{r['verifier']:<11}{r['cutoff']:>5} {r['cutoff_source']:<20}{syn:<20}{real:<24}{safe}")
    print("  (real-traffic numbers are per 1,000 questions, label-corrected)")
    print("\nUpper bound: cut-off AND threshold chosen on the real traffic itself (not a fair test)")
    for r in upper:
        detail = (f"c={r['cutoff']}, t={r['real_safe_threshold']}: {r['real_safe_correct']} correct, "
                  f"{r['real_safe_wrong']} wrong per 1,000" if r["cutoff"] != "none" else "none")
        print(f"  {r['model']:<7}{r['verifier']:<11}{r['cutoff_source']:<24}{detail}")
    print(f"\nWrote {SUMMARY}")


if __name__ == "__main__":
    {"score": score, "analyse": analyse}[sys.argv[1] if len(sys.argv) > 1 else "analyse"]()
