"""Check the automatic scores against hand labels.

Two automatic scores decide our results:
    scorer  answers_match() in simulate.py: is the cached answer a correct answer
            to the query? (decides every error count)
    judge   the Qwen LLM judge in verify.py: do the two questions have the same
            answer? (decides which hits are served)

Both should agree with a person reading the pair. Step 1 draws a sample of
served pairs (similarity >= 0.75) and writes a blind labelling sheet; step 2
compares the filled-in labels with both scores.

    python simulation/hand_labels.py sample   # writes data/hand_labels.csv (blind)
                                              # and results/hand_label_key.csv
    (fill in the "your_label" column: correct / wrong / unsure)
    python simulation/hand_labels.py check    # writes results/hand_label_check.json

The sample (seed 7) covers every query set:
The sample over-represents judge rejections and rare cases on purpose, so the
judge's agreement on it is not its overall rate; for that we also compare the
judge with the scorer on every served pair.

    paraphrase  all 5 pairs whose nearest cached question is not the intended one
                (the only paraphrase pairs where the scorer has to match aliases),
                15 the judge accepted and 10 it rejected
    each of the 5 adversarial categories: 5 pairs, plus every pair the judge
                accepted in comparative and entity (1 each)
"""
import csv
import json
import random
import sys

from simulate import DATA, OUT

SEED = 7
SHEET = DATA / "hand_labels.csv"
KEY = OUT / "hand_label_key.csv"
CHECK = OUT / "hand_label_check.json"
ADVERSARIAL = ["number", "entity", "comparative", "unit", "negation"]


def unique_pairs():
    """Served pairs (similarity >= 0.75) from both models, each pair once."""
    pairs = {}
    for model in ("minilm", "mpnet"):
        with (OUT / f"verdicts_{model}.csv").open(encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                pairs.setdefault((r["query"], r["nearest_question"]), r)
    return list(pairs.values())


def draw_sample(pairs):
    rng = random.Random(SEED)
    para = [r for r in pairs if r["query_set"] == "paraphrase"]
    picked = [r for r in para if r["is_target"] == "False"]
    target = [r for r in para if r["is_target"] == "True"]
    picked += rng.sample([r for r in target if r["judge_accept"] == "True"], 15)
    picked += rng.sample([r for r in target if r["judge_accept"] == "False"], 10)
    for cat in ADVERSARIAL:
        rows = [r for r in pairs if r["query_set"] == cat]
        forced = [r for r in rows if r["judge_accept"] == "True"] if cat in ("comparative", "entity") else []
        rest = [r for r in rows if r not in forced]
        picked += forced + rng.sample(rest, 5)
    rng.shuffle(picked)  # mix the sets so the sheet does not reveal them in order
    return picked


def sample():
    picked = draw_sample(unique_pairs())
    with SHEET.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label_id", "query", "reference_answer", "cached_question", "cached_answer",
                         "your_label (correct/wrong/unsure)", "your_note"])
        for i, r in enumerate(picked, 1):
            writer.writerow([i, r["query"], r["correct_answer"], r["nearest_question"], r["cached_answer"], "", ""])
    with KEY.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label_id", "query_set", "query_id", "nearest_id", "scorer_correct", "judge_accept"])
        for i, r in enumerate(picked, 1):
            writer.writerow([i, r["query_set"], r["query_id"], r["nearest_id"], r["answer_correct"], r["judge_accept"]])
    print(f"Wrote {SHEET} ({len(picked)} pairs to label) and {KEY}")


def kappa(a, b):
    """Cohen's kappa for two lists of True/False."""
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    return float("nan") if expected == 1 else (observed - expected) / (1 - expected)


def compare(name, auto, human, reference="human"):
    agree = sum(x == y for x, y in zip(auto, human))
    table = {f"{name}={'yes' if x else 'no'}, {reference}={'correct' if y else 'wrong'}":
             sum(1 for a, h in zip(auto, human) if (a, h) == (x, y))
             for x in (True, False) for y in (True, False)}
    return {"n": len(human), "agree": agree, "agreement": round(agree / len(human), 3),
            "kappa": round(kappa(auto, human), 3), "table": table}


def check():
    with SHEET.open(encoding="utf-8-sig", newline="") as f:
        labels = {r["label_id"]: r["your_label (correct/wrong/unsure)"].strip().lower() for r in csv.DictReader(f)}
    with KEY.open(encoding="utf-8", newline="") as f:
        key = {r["label_id"]: r for r in csv.DictReader(f)}
    bad = {i: v for i, v in labels.items() if v not in ("correct", "wrong", "unsure")}
    if bad:
        sys.exit(f"Labels must be correct / wrong / unsure; fix rows: {sorted(bad, key=int)}")
    used = [i for i in key if labels[i] != "unsure"]
    human = [labels[i] == "correct" for i in used]
    scorer = [key[i]["scorer_correct"] == "True" for i in used]
    judge = [key[i]["judge_accept"] == "True" for i in used]

    # The judge on every served pair, using the scorer as the reference.
    pairs = unique_pairs()
    j_all = [r["judge_accept"] == "True" for r in pairs]
    s_all = [r["answer_correct"] == "True" for r in pairs]

    result = {
        "labelled": len(labels),
        "unsure_excluded": len(labels) - len(used),
        "scorer_vs_human": compare("scorer", scorer, human),
        "judge_vs_human": compare("judge", judge, human),
        "judge_vs_scorer_all_served_pairs": compare("judge", j_all, s_all, reference="scorer"),
        "disagreements_scorer": [{"label_id": i, "query_set": key[i]["query_set"], "human": labels[i],
                                  "scorer_correct": key[i]["scorer_correct"]}
                                 for i in used if (key[i]["scorer_correct"] == "True") != (labels[i] == "correct")],
    }
    CHECK.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"Labelled {result['labelled']}, unsure (left out) {result['unsure_excluded']}")
    for name in ("scorer_vs_human", "judge_vs_human", "judge_vs_scorer_all_served_pairs"):
        r = result[name]
        print(f"\n{name}: agree {r['agree']}/{r['n']} ({r['agreement']:.1%}), kappa {r['kappa']}")
        for cell, count in r["table"].items():
            print(f"  {cell:<32} {count}")
    if result["disagreements_scorer"]:
        print("\nScorer disagreements:", result["disagreements_scorer"])
    print(f"\nWrote {CHECK}")


if __name__ == "__main__":
    {"sample": sample, "check": check}[sys.argv[1] if len(sys.argv) > 1 else "sample"]()
