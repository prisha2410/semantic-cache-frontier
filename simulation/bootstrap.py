"""Bootstrap: how much would the safe threshold move with a different sample of queries?

Our pipeline is deterministic, so running it again gives the same numbers. The
uncertainty comes from WHICH questions we happened to sample. We resample the
queries with replacement 1,000 times and redo the safe-threshold search of
costs.py on each resample.

    paraphrases  resampled by source question (both paraphrases of a question
                 together), since the two are not independent
    adversarial  resampled item by item within each category, keeping each
                 category's size, so the equal category weights stay the same

Output: results/bootstrap_safe_thresholds.csv, per model, verifier and D:
    the point estimate (costs.py), the share of resamples that have any safe
    threshold, and the median and 95% percentile range of the safe threshold
    and of the calls saved per 1,000 (0 when a resample has no safe threshold)

Run from the repo root:
    python simulation/bootstrap.py
"""
import numpy as np

from baseline import TFIDF_THRESHOLDS
from costs import BUDGET, CATEGORIES, D_SCENARIOS, R, VERIFIERS, accepts, load
from simulate import MODELS, OUT, THRESHOLDS, write_csv

B = 1000
SEED = 42


def matrices(by_set, verifier, thresholds):
    """Per query set: served[q, t] (hit and accepted) and wrong[q, t] (served and answer wrong)."""
    out = {}
    t = np.array(thresholds)
    for name, queries in by_set.items():
        sim = np.array([q["similarity"] for q in queries])
        ok = np.array([q["verdict"] is not None and accepts(verifier, q["verdict"]) if verifier != "none"
                       else True for q in queries])
        served = (sim[:, None] >= t[None, :]) & ok[:, None]
        correct = np.array([q["answer_correct"] for q in queries])
        out[name] = (served, served & ~correct[:, None])
    return out


def resample_indices(by_set, rng):
    """Row indices for one bootstrap resample of every query set."""
    idx = {}
    para = by_set["paraphrase"]
    groups = {}
    for i, q in enumerate(para):
        groups.setdefault(q["query_id"].split("_p")[0], []).append(i)
    keys = list(groups)
    picked = rng.integers(0, len(keys), len(keys))
    idx["paraphrase"] = np.array([i for k in picked for i in groups[keys[k]]])
    for cat in CATEGORIES:
        n = len(by_set[cat])
        idx[cat] = rng.integers(0, n, n)
    return idx


def safe_point(mats, idx, d):
    """Safe threshold index and calls saved for one resample (point estimate mode)."""
    served_p, wrong_p = mats["paraphrase"]
    p = idx["paraphrase"]
    served_rate = served_p[p].mean(axis=0)
    wrong = R * wrong_p[p].mean(axis=0)
    wrong = wrong + d * np.mean([mats[c][1][idx[c]].mean(axis=0) for c in CATEGORIES], axis=0)
    ok = np.flatnonzero(wrong <= BUDGET)
    if ok.size == 0:
        return None, 0.0
    best = ok[np.argmax(served_rate[ok])]  # first maximum = lowest threshold among ties
    return best, R * served_rate[best]


def main():
    rng = np.random.default_rng(SEED)
    data = {m: load(m) for m in MODELS + ["tfidf"]}
    # The same resamples for every model and verifier, so they are compared on equal footing.
    resamples = [resample_indices(data[MODELS[0]], rng) for _ in range(B)]
    full = {name: np.arange(len(q)) for name, q in data[MODELS[0]].items()}

    rows = []
    for model, verifiers, thresholds in [(m, VERIFIERS, THRESHOLDS) for m in MODELS] + \
                                        [("tfidf", ["none"], TFIDF_THRESHOLDS)]:
        for verifier in verifiers:
            mats = matrices(data[model], verifier, thresholds)
            for d in D_SCENARIOS:
                point_t, point_saved = safe_point(mats, full, d)
                ts, saved = [], []
                for idx in resamples:
                    t_i, s = safe_point(mats, idx, d)
                    saved.append(s)
                    if t_i is not None:
                        ts.append(thresholds[t_i])
                rows.append({
                    "model": model, "verifier": verifier, "d": d,
                    "point_safe_threshold": thresholds[point_t] if point_t is not None else "none",
                    "point_calls_saved": round(point_saved, 1),
                    "share_with_safe_threshold": len(ts) / B,
                    "threshold_median": np.median(ts) if ts else "",
                    "threshold_lo": np.percentile(ts, 2.5) if ts else "",
                    "threshold_hi": np.percentile(ts, 97.5) if ts else "",
                    "saved_median": round(float(np.median(saved)), 1),
                    "saved_lo": round(float(np.percentile(saved, 2.5)), 1),
                    "saved_hi": round(float(np.percentile(saved, 97.5)), 1),
                })
    write_csv(OUT / "bootstrap_safe_thresholds.csv", rows)

    print(f"Bootstrap of the safe threshold ({B} resamples, seed {SEED}); ranges are 95% percentile intervals")
    print(f"  {'model':<7}{'verifier':<9}{'D':>4}  {'point':<12}{'has safe t':>10}  "
          f"{'threshold range':<17}{'calls saved per 1,000':<22}")
    for r in rows:
        point = f"{r['point_safe_threshold']}" + (f" / {r['point_calls_saved']:.0f}" if r["point_calls_saved"] else "")
        t_range = f"{r['threshold_lo']:.2f}–{r['threshold_hi']:.2f}" if r["threshold_median"] != "" else "–"
        print(f"  {r['model']:<7}{r['verifier']:<9}{r['d']:>4}  {point:<12}{r['share_with_safe_threshold']:>9.0%}  "
              f"{t_range:<17}{r['saved_median']:.0f} ({r['saved_lo']:.0f}–{r['saved_hi']:.0f})")
    print(f"\nWrote {OUT / 'bootstrap_safe_thresholds.csv'}")


if __name__ == "__main__":
    main()
