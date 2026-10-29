"""Build the final paraphrase set from the reviewed candidates.

Inputs (all in data/):
    paraphrase_candidates.jsonl   model candidates (generate + retry rounds)
    paraphrase_review.csv         review verdict per candidate (accept / reject / ...)
    paraphrases_manual.tsv        hand-written paraphrases: <id> TAB <text>
    excluded_ids.txt              questions left out of the experiment

Output:
    data/paraphrases.json         per question: the cleaned question, its answer,
                                  and exactly 2 paraphrases tagged "model" or "manual"

Run from the repo root:
    python paraphrase/build_paraphrases.py
"""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
N_PER_QUESTION = 2


def main():
    candidates = {}
    with (DATA / "paraphrase_candidates.jsonl").open(encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            candidates[record["id"]] = record
    excluded = set((DATA / "excluded_ids.txt").read_text(encoding="utf-8").split())

    # Accepted model paraphrases, in candidate order.
    accepted = defaultdict(list)
    with (DATA / "paraphrase_review.csv").open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row["review"] == "accept":
                cand = candidates[row["id"]]["candidates"][int(row["cand"])]
                assert cand["text"] == row["candidate"], f"candidate text mismatch for {row['id']}"
                accepted[row["id"]].append({"text": cand["text"], "source": "model",
                                            "candidate_index": int(row["cand"])})

    # Hand-written paraphrases fill the gaps.
    manual = defaultdict(list)
    with (DATA / "paraphrases_manual.tsv").open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                qid, text = line.rstrip("\n").split("\t")
                manual[qid].append({"text": text, "source": "manual"})

    output, problems = [], []
    for qid, record in candidates.items():
        if qid in excluded:
            continue
        paraphrases = accepted[qid] + manual[qid]
        if len(paraphrases) != N_PER_QUESTION:
            problems.append(f"{qid}: {len(paraphrases)} paraphrases")
        output.append({"id": qid, "question": record["question"],
                       "ground_truth_answer": record["ground_truth_answer"],
                       "paraphrases": paraphrases})

    unused_manual = sorted(set(manual) - {o["id"] for o in output})
    if unused_manual:
        problems.append(f"manual paraphrases for excluded/unknown ids: {unused_manual}")
    if problems:
        raise SystemExit("Not written:\n  " + "\n  ".join(problems))

    out_file = DATA / "paraphrases.json"
    out_file.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    sources = Counter(p["source"] for o in output for p in o["paraphrases"])
    print(f"{len(output)} questions, {sum(sources.values())} paraphrases "
          f"({sources['model']} model, {sources['manual']} manual)")
    print(f"Wrote {out_file}")


if __name__ == "__main__":
    main()
