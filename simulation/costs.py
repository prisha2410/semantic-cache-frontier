"""Step 8b: cost per 1,000 queries and the safe threshold, with and without verifiers.

A cache hit is SERVED only if similarity >= threshold AND the verifier accepts
the pair; otherwise the query goes to the LLM (a call, but no wrong answer).

Verifiers: none, rule (rule_guard.py), judge (Qwen2.5-1.5B), both (rule AND judge).

Traffic model per 1,000 queries (illustrative assumptions, not measured):
    R = 300 genuine rephrasings of a cached question
    D = 10, 50 or 100 adversarial near-duplicates, split equally over the 5 categories
    calls saved   = R * paraphrase served rate
    wrong answers = R * paraphrase error rate + D * mean error rate over the 5 categories

Safe threshold: the threshold with the most calls saved while wrong answers <= 1
per 1,000 queries.
    point         uses the measured error rates
    conservative  uses the upper end of each 95% Wilson interval instead, because
                  0 errors in 30-50 items does not show the true rate is 0

Inputs:  results/nearest_<model>.csv, results/verdicts_<model>.csv
Outputs: results/costs.csv, results/safe_thresholds.csv,
         results/figures/verifier_frontier.png

Run from the repo root:
    python simulation/costs.py
"""
import csv

from plots import FIGS, INK, INK_2, MODEL_STYLE, MUTED, SURFACE, plt, style_axes
from simulate import MODELS, OUT, QUERY_SETS, THRESHOLDS, wilson, write_csv

VERIFIERS = ["none", "rule", "judge", "both"]
CATEGORIES = QUERY_SETS[1:]  # number, entity, comparative, unit, negation
R = 300
D_SCENARIOS = [10, 50, 100]
BUDGET = 1.0  # wrong answers per 1,000 queries


def accepts(verifier, row):
    if verifier == "none":
        return True
    rule = row["rule_accept"] == "True"
    judge = row["judge_accept"] == "True"
    return {"rule": rule, "judge": judge, "both": rule and judge}[verifier]


def load(model):
    """Every query with its similarity, answer check and (if similarity >= 0.75) verdicts."""
    with (OUT / f"verdicts_{model}.csv").open(encoding="utf-8", newline="") as f:
        verdicts = {(r["query_set"], r["query_id"]): r for r in csv.DictReader(f)}
    with (OUT / f"nearest_{model}.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    by_set = {}
    for r in rows:
        sim = float(r["similarity"])
        v = verdicts.get((r["query_set"], r["query_id"]))
        assert v is not None or sim < THRESHOLDS[0], f"missing verdict for {r['query_id']}"
        by_set.setdefault(r["query_set"], []).append({
            "similarity": sim,
            "answer_correct": r["answer_correct"] == "True",
            "verdict": v,
        })
    return by_set


def rates(queries, verifier, t):
    """Served count and wrong-answer count at threshold t."""
    served = [q for q in queries if q["similarity"] >= t and accepts(verifier, q["verdict"])]
    wrong = sum(not q["answer_correct"] for q in served)
    return len(served), wrong


def cost_rows(model, by_set):
    out = []
    for verifier in VERIFIERS:
        for t in THRESHOLDS:
            row = {"model": model, "verifier": verifier, "threshold": t}
            para = by_set["paraphrase"]
            served, wrong = rates(para, verifier, t)
            row["paraphrase_n"] = len(para)
            row["paraphrase_served_rate"] = served / len(para)
            row["paraphrase_error_rate"] = wrong / len(para)
            row["paraphrase_error_hi"] = wilson(wrong, len(para))[1]
            for cat in CATEGORIES:
                qs = by_set[cat]
                _, wrong = rates(qs, verifier, t)
                row[f"{cat}_error_rate"] = wrong / len(qs)
                row[f"{cat}_error_hi"] = wilson(wrong, len(qs))[1]
            row["mean_error_rate"] = sum(row[f"{c}_error_rate"] for c in CATEGORIES) / len(CATEGORIES)
            row["mean_error_hi"] = sum(row[f"{c}_error_hi"] for c in CATEGORIES) / len(CATEGORIES)
            row["calls_saved_per_1000"] = R * row["paraphrase_served_rate"]
            for d in D_SCENARIOS:
                row[f"wrong_per_1000_d{d}"] = R * row["paraphrase_error_rate"] + d * row["mean_error_rate"]
                row[f"wrong_per_1000_d{d}_conservative"] = R * row["paraphrase_error_hi"] + d * row["mean_error_hi"]
            out.append(row)
    return out


def safe_thresholds(costs):
    """Per model, verifier and D: the safe threshold under both modes.

    For the point-safe threshold we also report the conservative wrong-answer
    bound at that same threshold, so the gap between the two is visible.
    """
    out = []
    for model in MODELS:
        for verifier in VERIFIERS:
            series = [r for r in costs if r["model"] == model and r["verifier"] == verifier]
            for d in D_SCENARIOS:
                point, cons = f"wrong_per_1000_d{d}", f"wrong_per_1000_d{d}_conservative"
                row = {"model": model, "verifier": verifier, "d": d}
                for mode, col in (("point", point), ("conservative", cons)):
                    ok = [r for r in series if r[col] <= BUDGET]
                    best = max(ok, key=lambda r: (r["calls_saved_per_1000"], -r["threshold"]), default=None)
                    row[f"{mode}_safe_threshold"] = best["threshold"] if best else "none"
                    row[f"{mode}_calls_saved_per_1000"] = round(best["calls_saved_per_1000"], 1) if best else 0.0
                    row[f"{mode}_wrong_per_1000"] = round(best[col], 2) if best else ""
                    if mode == "point":
                        row["point_conservative_wrong_per_1000"] = round(best[cons], 2) if best else ""
                out.append(row)
    return out


def figure(costs):
    """One panel per verifier: mean near-duplicate error vs paraphrase served rate."""
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.4), facecolor=SURFACE, sharey=True)
    titles = {"none": "no verifier", "rule": "rule guard", "judge": "LLM judge", "both": "rule guard + LLM judge"}
    for k, (ax, verifier) in enumerate(zip(axes, VERIFIERS)):
        style_axes(ax, titles[verifier], "paraphrase served rate (savings)",
                   "mean error rate, 5 categories" if k == 0 else "")
        ax.set_xlim(-0.02, 1.02)
        # Budget lines: with no paraphrase errors, wrong <= 1 per 1,000 means mean error <= 1/D.
        for d in D_SCENARIOS:
            ax.axhline(BUDGET / d, color=MUTED, linewidth=0.8, linestyle=(0, (3, 3)))
        for model, st in MODEL_STYLE.items():
            s = [r for r in costs if r["model"] == model and r["verifier"] == verifier]
            ax.plot([r["paraphrase_served_rate"] for r in s], [r["mean_error_rate"] for r in s],
                    color=st["color"], linewidth=2, label=st["label"])
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left")
    fig.suptitle("Verifiers move the frontier down: savings vs errors on near-duplicates "
                 "(thresholds 0.75–0.98; equal category weights)",
                 color=INK, fontsize=12, x=0.01, ha="left")
    fig.text(0.01, 0.01, "Dashed lines (top to bottom): the most mean error that keeps wrong answers within "
             "1 per 1,000 queries for D = 10, 50, 100 near-duplicates per 1,000 (illustrative).",
             color=INK_2, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    return fig


def main():
    costs = []
    for model in MODELS:
        costs += cost_rows(model, load(model))
    write_csv(OUT / "costs.csv", costs)
    safe = safe_thresholds(costs)
    write_csv(OUT / "safe_thresholds.csv", safe)

    FIGS.mkdir(parents=True, exist_ok=True)
    fig = figure(costs)
    path = FIGS / "verifier_frontier.png"
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)

    print(f"Safe threshold (most calls saved with <= {BUDGET:g} wrong per 1,000; R = {R}):")
    print(f"  {'model':<7}{'verifier':<9}{'D':>4}   {'point: t, saved, wrong (upper bound)':<40}conservative: t, saved")
    for s in safe:
        if s["point_safe_threshold"] == "none":
            point = "none"
        else:
            point = (f"t={s['point_safe_threshold']:<5} saved {s['point_calls_saved_per_1000']:5.1f}  "
                     f"wrong {s['point_wrong_per_1000']:.2f} ({s['point_conservative_wrong_per_1000']:.2f})")
        cons = ("none" if s["conservative_safe_threshold"] == "none" else
                f"t={s['conservative_safe_threshold']:<5} saved {s['conservative_calls_saved_per_1000']:5.1f}")
        print(f"  {s['model']:<7}{s['verifier']:<9}{s['d']:>4}   {point:<40}{cons}")
    print(f"\nWrote {OUT / 'costs.csv'} ({len(costs)} rows), {OUT / 'safe_thresholds.csv'}, {path}")


if __name__ == "__main__":
    main()
