"""Figures for the report, drawn from results/sweep.csv.

    results/figures/frontier.png         cost-correctness frontier: adversarial error
                                         rate vs paraphrase hit rate (savings), one panel
                                         per category, one line per model, plus the
                                         TF-IDF baseline (results/sweep_tfidf.csv)
    results/figures/threshold_sweep.png  paraphrase hit rate and each category's error
                                         rate vs threshold, with 95% Wilson intervals

Run from the repo root:
    python simulation/plots.py
"""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files only, no window
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
FIGS = RESULTS / "figures"

# Reference palette (dataviz skill), categorical slots 1-2; validated as a pair.
MODEL_STYLE = {
    "minilm": {"color": "#2a78d6", "label": "MiniLM (small)"},
    "mpnet": {"color": "#eb6834", "label": "MPNet (stronger)"},
}
# Baseline, slot 3 (the first three slots validate as a set). Frontier figure only:
# its thresholds run 0.30-0.98, so it does not share the threshold-sweep x-axis.
BASELINE_STYLE = {"tfidf": {"color": "#1baf7a", "label": "TF-IDF baseline (word overlap)"}}
INK, INK_2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
CATEGORIES = ["number", "entity", "comparative", "unit", "negation"]
MARKED_THRESHOLDS = [0.85, 0.90, 0.95]


def load_sweep():
    rows = []
    for name in ("sweep.csv", "sweep_tfidf.csv"):
        with (RESULTS / name).open(encoding="utf-8", newline="") as f:
            rows += list(csv.DictReader(f))
    data = {}
    for r in rows:
        key = (r["model"], r["query_set"])
        data.setdefault(key, []).append({k: (float(v) if k not in ("model", "query_set") else v)
                                         for k, v in r.items()})
    for series in data.values():
        series.sort(key=lambda r: r["threshold"])
    return data


def style_axes(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=8)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.set_ylim(-0.02, 1.02)


def frontier(data):
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5), facecolor=SURFACE)
    axes = axes.ravel()
    for ax, cat in zip(axes, CATEGORIES):
        n = int(data[("minilm", cat)][0]["n"])
        style_axes(ax, f"{cat}  (n = {n})", "paraphrase hit rate (savings)", "error rate on near-duplicates")
        ax.set_xlim(-0.02, 1.02)
        for model, st in {**MODEL_STYLE, **BASELINE_STYLE}.items():
            para = data[(model, "paraphrase")]
            adv = data[(model, cat)]
            x = [p["hit_rate"] for p in para]
            y = [a["error_rate"] for a in adv]
            ax.plot(x, y, color=st["color"], linewidth=2, label=st["label"])
            if model in BASELINE_STYLE:
                continue
            for t in MARKED_THRESHOLDS:
                i = next(k for k, p in enumerate(para) if abs(p["threshold"] - t) < 1e-9)
                ax.plot(x[i], y[i], "o", markersize=6, color=st["color"],
                        markeredgecolor=SURFACE, markeredgewidth=1.5)
    # Direct labels where the baseline and the models are furthest apart (entity panel).
    axes[1].text(0.43, 0.50, "TF-IDF baseline", color=INK_2, fontsize=8, ha="right")
    axes[1].text(0.58, 0.05, "both embedding models", color=INK_2, fontsize=8, ha="center")
    legend_ax = axes[-1]
    legend_ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    legend_ax.legend(handles, labels, loc="upper left", frameon=False, fontsize=10, labelcolor=INK)
    legend_ax.text(0.0, 0.62,
                   "Each line traces thresholds 0.50 (right) to 0.98 (left);\n"
                   "the TF-IDF line runs 0.30 to 0.98.\n"
                   "Dots mark t = 0.95, 0.90, 0.85 (left to right).\n"
                   "Lower-right is better: more savings, fewer errors.\n"
                   "Error rate = share of near-duplicate queries that\n"
                   "received a wrong cached answer.",
                   transform=legend_ax.transAxes, color=INK_2, fontsize=9, va="top")
    fig.suptitle("Cost–correctness frontier: savings on paraphrases vs errors on near-duplicates",
                 color=INK, fontsize=13, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def threshold_sweep(data):
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5), facecolor=SURFACE, sharex=True)
    axes = axes.ravel()
    panels = [("paraphrase", "hit_rate", "paraphrase hit rate", "hit rate (higher = more savings)")]
    panels += [(c, "error_rate", f"{c} error rate", "error rate (lower = safer)") for c in CATEGORIES]
    for k, (ax, (qset, metric, title, ylabel)) in enumerate(zip(axes, panels)):
        n = int(data[("minilm", qset)][0]["n"])
        style_axes(ax, f"{title}  (n = {n})", "similarity threshold" if k >= 3 else "", ylabel)
        for model, st in MODEL_STYLE.items():
            s = data[(model, qset)]
            t = [r["threshold"] for r in s]
            ax.fill_between(t, [r[f"{metric}_lo"] for r in s], [r[f"{metric}_hi"] for r in s],
                            color=st["color"], alpha=0.15, linewidth=0)
            ax.plot(t, [r[metric] for r in s], color=st["color"], linewidth=2, label=st["label"])
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower left")
    # Direct labels where the lines separate (comparative panel).
    axes[3].text(0.885, 0.42, MODEL_STYLE["minilm"]["label"], color=INK_2, fontsize=8, ha="right")
    axes[3].text(0.925, 0.80, MODEL_STYLE["mpnet"]["label"], color=INK_2, fontsize=8, ha="left")
    fig.suptitle("Threshold sweep (0.50–0.98); shaded bands are 95% Wilson intervals",
                 color=INK, fontsize=13, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    data = load_sweep()
    for name, fig in (("frontier", frontier(data)), ("threshold_sweep", threshold_sweep(data))):
        path = FIGS / f"{name}.png"
        fig.savefig(path, dpi=160, facecolor=SURFACE)
        plt.close(fig)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
