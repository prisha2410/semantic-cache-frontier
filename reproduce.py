"""Regenerate the main result with one command.

    python reproduce.py

Runs, in order:
    simulation/simulate.py           threshold sweep for both embedding models
    simulation/baseline.py           exact-match and TF-IDF baselines
    simulation/plots.py              frontier.png, threshold_sweep.png
    simulation/verify.py             rule guard + LLM judge verdicts (from judge_cache.jsonl:
                                     no GPU needed unless a new pair appears)
    simulation/costs.py              cost per 1,000 queries and safe thresholds,
                                     verifier_frontier.png
    simulation/bootstrap.py          spread of the safe threshold over resampled query sets
    simulation/hand_labels.py check  answer scorer and judge vs our hand labels

It starts from the committed embeddings (data/embeddings/), paraphrases and
adversarial set; rebuilding those needs the GPU scripts in paraphrase/ and
embeddings/.

Before running, it keeps a copy of the main result files; afterwards it checks
the regenerated files give the same numbers (to a tolerance of 1e-6, since
floating-point sums can differ in the last digits between machines).
"""
import csv
import json
import math
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
RESULTS = REPO / "results"

STEPS = [
    ["simulation/simulate.py"],
    ["simulation/baseline.py"],
    ["simulation/plots.py"],
    ["simulation/verify.py"],
    ["simulation/costs.py"],
    ["simulation/bootstrap.py"],
    ["simulation/hand_labels.py", "check"],
]
CHECKED = ["sweep.csv", "sweep_tfidf.csv", "baseline_exact.csv", "costs.csv",
           "safe_thresholds.csv", "bootstrap_safe_thresholds.csv", "hand_label_check.json"]


def read(path):
    """Parsed content: rows of a CSV, or a JSON object."""
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.reader(f))


def same(a, b):
    """Equal, treating numbers as equal within 1e-6."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    try:
        x, y = float(a), float(b)
    except (TypeError, ValueError):
        return a == b
    return (math.isnan(x) and math.isnan(y)) or math.isclose(x, y, rel_tol=1e-6, abs_tol=1e-9)


def main():
    before = {name: read(RESULTS / name) for name in CHECKED if (RESULTS / name).exists()}
    start = time.time()
    for step in STEPS:
        print(f"\n=== python {' '.join(step)}", flush=True)
        subprocess.run([sys.executable, *step], cwd=REPO, check=True)

    print(f"\n=== Reproducibility check ({time.time() - start:.0f}s in total)")
    all_same = True
    for name in CHECKED:
        if name not in before:
            print(f"  {name:<30} new (nothing committed to compare with)")
            continue
        ok = same(before[name], read(RESULTS / name))
        all_same &= ok
        print(f"  {name:<30} {'same numbers' if ok else 'DIFFERENT'}")
    print("\nAll main results reproduced." if all_same else
          "\nSome results differ from the committed ones: see `git diff results/`.")


if __name__ == "__main__":
    main()
