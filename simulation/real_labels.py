"""Check the answer scorer on REAL traffic against hand labels.

On real traffic (real_traffic.py) the cached answer comes from a different
TriviaQA row, so the same answer is often written differently ("Disney" vs
"Walt Disney", "Tasman" vs "Tasman Sea"). Our 57 hand labels never covered
that case: in our own data the cached answer was always copied exactly.

Two scorers are compared with a person:
    strict   answers_match() in simulate.py (the one used everywhere so far)
    lenient  strict, OR one answer appears inside the other as whole words
             after normalising ("tasman" inside "tasman sea")

    python simulation/real_labels.py sample   # writes data/real_traffic_labels.csv (blind)
                                              # and results/real_traffic_label_key.csv
    (fill in the "your_label" column: correct / wrong / unsure)
    python simulation/real_labels.py check    # writes results/real_traffic_label_check.json

Sample (seed 11): served pairs (similarity >= 0.50, the lowest threshold we use) from both models, each pair
once, in three groups of 20:
    A  strict says wrong, lenient says correct   (the disputed cases)
    B  both say wrong                            (claimed real errors)
    C  both say correct                          (claimed correct answers)
Because each group is sampled at its own rate, the check reports each group's
error rate as labelled, and from those a label-corrected count of wrong answers
per 1,000 queries at the settings in real_traffic.py.
"""
import csv
import json
import random
import re
import sys

from real_traffic import PAIRS, RECOMMENDED, accepts
from simulate import DATA, OUT, normalize_answer, wilson

SEED = 11
PER_GROUP = 20
MIN_SIMILARITY = 0.50
SHEET = DATA / "real_traffic_labels.csv"
KEY = OUT / "real_traffic_label_key.csv"
CHECK = OUT / "real_traffic_label_check.json"
SETTINGS = [("minilm", "none", 0.75), ("minilm", "none", 0.90), ("minilm", "both", 0.90),
            ("minilm", "both", 0.95), ("mpnet", "both", 0.90), ("mpnet", "both", 0.95)] + RECOMMENDED


def contains(a, b):
    """True if one normalised answer appears inside the other as whole words."""
    a, b = normalize_answer(a), normalize_answer(b)
    if not a or not b:
        return False
    return bool(re.search(rf"\b{re.escape(a)}\b", b) or re.search(rf"\b{re.escape(b)}\b", a))


def group(row):
    strict = row["answer_correct"] == "True"
    lenient = strict or contains(row["cached_answer"], row["query_answer"])
    return "C" if strict else ("A" if lenient else "B")


def load_rows():
    with PAIRS.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def sample():
    pairs = {}
    for r in load_rows():
        if float(r["similarity"]) >= MIN_SIMILARITY:
            pairs.setdefault((r["query"], r["nearest_question"]), r)
    rng = random.Random(SEED)
    picked = []
    for g in ("A", "B", "C"):
        members = [r for r in pairs.values() if group(r) == g]
        picked += rng.sample(members, min(PER_GROUP, len(members)))
    rng.shuffle(picked)
    with SHEET.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label_id", "query", "reference_answer", "cached_question", "cached_answer",
                         "your_label (correct/wrong/unsure)", "your_note"])
        for i, r in enumerate(picked, 1):
            writer.writerow([i, r["query"], r["query_answer"], r["nearest_question"], r["cached_answer"], "", ""])
    with KEY.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label_id", "group", "query_id", "nearest_id", "similarity"])
        for i, r in enumerate(picked, 1):
            writer.writerow([i, group(r), r["query_id"], r["nearest_id"], r["similarity"]])
    print(f"Wrote {SHEET} ({len(picked)} pairs to label) and {KEY}")


def kappa(a, b):
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    return float("nan") if expected == 1 else (observed - expected) / (1 - expected)


def check():
    with SHEET.open(encoding="utf-8-sig", newline="") as f:
        labels = {r["label_id"]: r["your_label (correct/wrong/unsure)"].strip().lower() for r in csv.DictReader(f)}
    with KEY.open(encoding="utf-8", newline="") as f:
        key = {r["label_id"]: r for r in csv.DictReader(f)}
    bad = sorted((i for i, v in labels.items() if v not in ("correct", "wrong", "unsure")), key=int)
    if bad:
        sys.exit(f"Labels must be correct / wrong / unsure; fix rows: {bad}")
    used = [i for i in key if labels[i] != "unsure"]
    human = [labels[i] == "correct" for i in used]
    strict = [key[i]["group"] == "C" for i in used]
    lenient = [key[i]["group"] in ("A", "C") for i in used]

    # Share of each group that the person judged wrong.
    wrong_rate = {}
    for g in ("A", "B", "C"):
        ids = [i for i in used if key[i]["group"] == g]
        k = sum(labels[i] == "wrong" for i in ids)
        lo, hi = wilson(k, len(ids))
        wrong_rate[g] = {"labelled": len(ids), "wrong": k, "rate": round(k / len(ids), 3) if ids else None,
                         "rate_lo": round(lo, 3), "rate_hi": round(hi, 3)}

    # Label-corrected wrong answers per 1,000: each served pair counts as wrong with
    # the probability its group was judged wrong.
    rows = load_rows()
    corrected = []
    for model, verifier, t in SETTINGS:
        pairs = [r for r in rows if r["model"] == model]
        served = [r for r in pairs if float(r["similarity"]) >= t and accepts(verifier, r)]
        counts = {g: sum(group(r) == g for r in served) for g in ("A", "B", "C")}
        est = sum(counts[g] * (wrong_rate[g]["rate"] or 0) for g in counts)
        corrected.append({"model": model, "verifier": verifier, "threshold": t,
                          "served_per_1000": round(1000 * len(served) / len(pairs), 1),
                          "wrong_per_1000_strict": round(1000 * (counts["A"] + counts["B"]) / len(pairs), 1),
                          "wrong_per_1000_lenient": round(1000 * counts["B"] / len(pairs), 1),
                          "wrong_per_1000_label_corrected": round(1000 * est / len(pairs), 1)})

    result = {
        "labelled": len(labels), "unsure_excluded": len(labels) - len(used),
        "strict_vs_human": {"agree": sum(x == y for x, y in zip(strict, human)), "n": len(used),
                            "kappa": round(kappa(strict, human), 3)},
        "lenient_vs_human": {"agree": sum(x == y for x, y in zip(lenient, human)), "n": len(used),
                             "kappa": round(kappa(lenient, human), 3)},
        "wrong_rate_by_group": wrong_rate,
        "wrong_per_1000": corrected,
    }
    CHECK.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"Labelled {result['labelled']}, unsure (left out) {result['unsure_excluded']}")
    for name in ("strict_vs_human", "lenient_vs_human"):
        r = result[name]
        print(f"  {name:<18} agree {r['agree']}/{r['n']}, kappa {r['kappa']}")
    print("\nShare judged wrong by you, per group:")
    for g, desc in (("A", "strict wrong, lenient correct"), ("B", "both wrong"), ("C", "both correct")):
        r = wrong_rate[g]
        print(f"  {g} ({desc:<30}) {r['wrong']}/{r['labelled']}")
    print("\nWrong answers per 1,000 real questions:")
    print(f"  {'model':<7}{'verifier':<9}{'t':>5}  {'served':>7}  {'strict':>7}  {'lenient':>8}  {'label-corrected':>16}")
    for r in corrected:
        print(f"  {r['model']:<7}{r['verifier']:<9}{r['threshold']:>5}  {r['served_per_1000']:>7}  "
              f"{r['wrong_per_1000_strict']:>7}  {r['wrong_per_1000_lenient']:>8}  {r['wrong_per_1000_label_corrected']:>16}")
    print(f"\nWrote {CHECK}")


if __name__ == "__main__":
    {"sample": sample, "check": check}[sys.argv[1] if len(sys.argv) > 1 else "sample"]()
