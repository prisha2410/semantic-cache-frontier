#!/usr/bin/env python3
"""Build data/source_questions.json from TriviaQA (closed-book / "nocontext").

This script is the single source of truth for HOW source questions are chosen.
The exploration notebook (notebooks/explore_triviaqa.ipynb) imports the same
functions, so what you inspect there is exactly what gets exported here.

Usage (from anywhere):
    python data/build_source_questions.py
    python data/build_source_questions.py --n 1000 --seed 42 --max-words 18

Outputs (next to this script by default):
    source_questions.json          list of {id, question, ground_truth_answer, source_metadata}
    source_questions_report.json   filter funnel + parameters (use for the data-source note)

Requires:  pip install datasets
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

DATASET_NAME = "mandarjoshi/trivia_qa"

# --------------------------------------------------------------------------- #
# Filter parameters
# --------------------------------------------------------------------------- #


@dataclass
class FilterConfig:
    min_words: int = 4           # drop fragments
    max_words: int = 20          # drop long, puzzle-style questions
    max_answer_words: int = 5    # keep short, checkable answers


# --------------------------------------------------------------------------- #
# Heuristics (all regex-based, so they are approximate: check them in the notebook)
# --------------------------------------------------------------------------- #

YEAR_RE = re.compile(r"(1[0-9]{3}|20[0-9]{2})")          # fullmatch on a digit run
DIGITS_RE = re.compile(r"\d+(?:[.,]\d+)?")
ROMAN_RE = re.compile(r"\b[IVXL]{2,}\b")                  # Super Bowl XX, World War II
COMPARATIVE_RE = re.compile(
    r"\b(first|last|largest|smallest|biggest|tallest|oldest|youngest|highest|lowest|"
    r"longest|shortest|fastest|slowest|deepest|heaviest|lightest|richest|best|worst|"
    r"most|least|more|less|larger|smaller|bigger|taller|older|younger|higher|lower|"
    r"longer|shorter|faster|slower|greater|earlier|earliest)\b",
    re.I,
)
UNIT_RE = re.compile(
    r"\b(miles?|kilomet(?:er|re)s?|km|met(?:er|re)s?|feet|foot|inch(?:es)?|yards?|"
    r"pounds?|kilograms?|kg|grams?|ounces?|tons?|tonnes?|degrees?|celsius|fahrenheit|"
    r"kelvin|lit(?:er|re)s?|gallons?|acres?|hectares?|mph|knots?|dollars?|euros?|"
    r"percent|calories)\b",
    re.I,
)
YES_NO_RE = re.compile(
    r"^(is|are|was|were|do|does|did|can|could|has|have|had|will|would|should)\b", re.I
)
TIME_SENSITIVE_RE = re.compile(
    r"\b(current|currently|latest|recent|recently|today|nowadays|at present|this year|"
    r"incumbent|reigning|so far|to date|as of)\b",
    re.I,
)
MULTI_ANSWER_RE = re.compile(
    r"\b(name (the|two|three|four|five)|which (two|three|four|five)|what are the|list)\b",
    re.I,
)

# Order matters: a row is reported under the FIRST stage it fails.
STAGES = [
    "answer_missing",
    "answer_too_long",
    "question_length",
    "time_sensitive",
    "multi_answer",
    "no_modifiable_feature",
]


def find_features(question: str) -> list[str]:
    """Which adversarial-friendly elements does the question contain?"""
    feats: list[str] = []
    digit_runs = DIGITS_RE.findall(question)
    if any(YEAR_RE.fullmatch(d.replace(",", "")) for d in digit_runs):
        feats.append("year")
    if any(not YEAR_RE.fullmatch(d.replace(",", "")) for d in digit_runs):
        feats.append("number")
    if ROMAN_RE.search(question):
        feats.append("roman_numeral")
    if COMPARATIVE_RE.search(question):
        feats.append("comparative")
    if UNIT_RE.search(question):
        feats.append("unit")
    if YES_NO_RE.search(question.strip()):
        feats.append("yes_no")
    tokens = re.findall(r"[A-Za-z][\w'’\-]*", question)
    # Capitalised word that is not the first word, ignoring the pronoun "I".
    if any(t[0].isupper() and t != "I" for t in tokens[1:]):
        feats.append("entity")  # nearly always true -> a weak filter by design
    return feats


def evaluate(record: dict, cfg: FilterConfig) -> tuple[str | None, list[str]]:
    """Return (reject_reason or None, features)."""
    q = (record.get("question") or "").strip()
    ans = record.get("answer") or {}
    value = (ans.get("value") or "").strip()
    feats = find_features(q) if q else []

    if not value:
        return "answer_missing", feats
    if len(value.split()) > cfg.max_answer_words:
        return "answer_too_long", feats
    n_words = len(q.split())
    if n_words < cfg.min_words or n_words > cfg.max_words:
        return "question_length", feats
    if TIME_SENSITIVE_RE.search(q):
        return "time_sensitive", feats
    if MULTI_ANSWER_RE.search(q):
        return "multi_answer", feats
    if not feats:
        return "no_modifiable_feature", feats
    return None, feats


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def load_triviaqa(config: str = "unfiltered.nocontext", split: str = "train"):
    """Yield minimal records from HuggingFace (import is lazy so tests can skip it)."""
    from datasets import load_dataset

    ds = load_dataset(DATASET_NAME, config, split=split)
    for i, row in enumerate(ds):
        yield {
            "row_index": i,
            "question": row.get("question"),
            "question_id": row.get("question_id"),
            "question_source": row.get("question_source"),
            "answer": row.get("answer") or {},
        }


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #


def _norm_question(q: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", "", q.lower())).strip()


def select(records, cfg: FilterConfig, n: int, seed: int,
           config_name: str = "unfiltered.nocontext", split: str = "train"):
    """Filter -> dedupe -> seeded sample. Returns (output_records, report)."""
    reasons: Counter = Counter()
    passed: list[dict] = []
    total = 0
    for rec in records:
        total += 1
        reason, feats = evaluate(rec, cfg)
        if reason:
            reasons[reason] += 1
            continue
        passed.append({**rec, "features": feats})

    seen: set[str] = set()
    deduped: list[dict] = []
    for rec in passed:
        key = _norm_question(rec["question"])
        if key in seen:
            reasons["duplicate_question"] += 1
            continue
        seen.add(key)
        deduped.append(rec)

    if n > len(deduped):
        raise ValueError(
            f"Asked for n={n} but only {len(deduped)} questions survive the filters. "
            "Loosen the filters (e.g. --max-words) or lower --n."
        )

    rng = random.Random(seed)
    chosen_idx = sorted(rng.sample(range(len(deduped)), n))  # keep dataset order
    chosen = [deduped[i] for i in chosen_idx]

    output = []
    for k, rec in enumerate(chosen, start=1):
        ans = rec["answer"]
        output.append({
            "id": f"q{k:04d}",
            "question": rec["question"].strip(),
            "ground_truth_answer": (ans.get("value") or "").strip(),
            "source_metadata": {
                "dataset": DATASET_NAME,
                "config": config_name,
                "split": split,
                "row_index": rec["row_index"],
                "question_id": rec["question_id"],
                "question_source": rec["question_source"],
                "answer_type": ans.get("type"),
                "normalized_value": ans.get("normalized_value"),
                "aliases": ans.get("aliases") or [],
                "normalized_aliases": ans.get("normalized_aliases") or [],
                "modifiable_features": rec["features"],
            },
        })

    # Funnel: how many rows are left after each stage.
    remaining = total
    funnel = [{"stage": "total_rows", "remaining": remaining}]
    for stage in STAGES + ["duplicate_question"]:
        remaining -= reasons.get(stage, 0)
        funnel.append({"stage": f"after_{stage}", "remaining": remaining})
    funnel.append({"stage": "sampled", "remaining": n})

    feature_counts = Counter(f for o in output for f in o["source_metadata"]["modifiable_features"])
    type_counts = Counter(o["source_metadata"]["answer_type"] for o in output)
    report = {
        "dataset": DATASET_NAME,
        "config": config_name,
        "split": split,
        "seed": seed,
        "n": n,
        "filter_config": asdict(cfg),
        "rejections_by_first_failed_stage": dict(reasons),
        "funnel": funnel,
        "selected_feature_counts": dict(feature_counts),
        "selected_answer_type_counts": {str(k): v for k, v in type_counts.items()},
    }
    return output, report


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main() -> None:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="unfiltered.nocontext")
    p.add_argument("--split", default="train")
    p.add_argument("--n", type=int, default=800, help="number of source questions to export")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-words", type=int, default=FilterConfig.min_words)
    p.add_argument("--max-words", type=int, default=FilterConfig.max_words)
    p.add_argument("--max-answer-words", type=int, default=FilterConfig.max_answer_words)
    p.add_argument("--out", type=Path, default=here / "source_questions.json")
    p.add_argument("--report", type=Path, default=here / "source_questions_report.json")
    args = p.parse_args()

    cfg = FilterConfig(args.min_words, args.max_words, args.max_answer_words)
    print(f"Loading {DATASET_NAME} [{args.config}] split={args.split} ...")
    records = load_triviaqa(args.config, args.split)
    output, report = select(records, cfg, args.n, args.seed, args.config, args.split)

    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nFilter funnel:")
    for step in report["funnel"]:
        print(f"  {step['stage']:<32} {step['remaining']:>8}")
    print(f"\nFeature counts in the {args.n} selected questions:")
    for feat, c in sorted(report["selected_feature_counts"].items(), key=lambda x: -x[1]):
        print(f"  {feat:<16} {c:>6}")
    print(f"\nWrote {args.out}\nWrote {args.report}")


if __name__ == "__main__":
    main()
