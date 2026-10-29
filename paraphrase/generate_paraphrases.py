"""Generate paraphrase candidates for all source questions (minus exclusions).

For each question we ask Qwen2.5-1.5B-Instruct for 5 rewrites, then mark each
rewrite with an automatic filter result. Nothing is thrown away: rejected
candidates are kept with their reason, so the review can see everything.

Output (one JSON object per line, so the run can be stopped and resumed):
    data/paraphrase_candidates.jsonl
    data/paraphrase_candidates_meta.json   model, prompt, settings, versions, timing

Run from the repo root:
    python paraphrase/generate_paraphrases.py            # all questions
    python paraphrase/generate_paraphrases.py --limit 20 # quick check on the first 20

If the run stops, start it again with the same command: questions already in
the output file are skipped.
"""
import argparse
import json
import platform
import re
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
OUT_FILE = DATA / "paraphrase_candidates.jsonl"
META_FILE = DATA / "paraphrase_candidates_meta.json"

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 300

PROMPT = """Rewrite the question below in 5 different ways.
Rules:
- Keep the exact meaning: the correct answer must stay the same.
- Ask for the same thing as the original (the same person, object or place being asked about).
- Keep every name, number, date and quoted phrase exactly as written.
- Do not include the answer or any hint about it.
- Do not write answer options, lists, or phrases like "which of the following" or "which of these".
- Each rewrite must be a question ending with "?".
- Write one rewrite per line, with no numbering and no extra text.

Question: {question}"""

LIST_PHRASES = re.compile(r"\b(of the following|of these)\b", re.I)


# --------------------------------------------------------------------------- #
# Cleaning and filters (same rules as the 10-question test)
# --------------------------------------------------------------------------- #

def clean(question):
    """Undo CSV-style quoting: '"Say ""hi"" to whom?"' -> 'Say "hi" to whom?'."""
    if question.startswith('"') and question.endswith('"') and '""' in question:
        question = question[1:-1].replace('""', '"')
    return question


def normalize(text):
    """Lowercase and keep only letters/digits, so 'The Winter's Tale' -> 'the winter s tale'."""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def contains_phrase(text, phrase):
    """True if `phrase` appears in `text` as whole words (after normalizing both)."""
    phrase = normalize(phrase)
    return bool(phrase) and re.search(rf"\b{re.escape(phrase)}\b", normalize(text)) is not None


def answer_strings(q):
    """The answer plus its aliases, skipping very short ones (e.g. 'Ra') that match too easily."""
    strings = [q["ground_truth_answer"]] + q["source_metadata"]["aliases"]
    return [s for s in strings if len(normalize(s)) >= 3]


def reject_reason(candidate, original, q):
    """Return why a candidate is rejected, or None if it passes every check."""
    if not candidate.endswith("?"):
        return "not a question"
    if normalize(candidate) == normalize(original):
        return "same as original"
    if LIST_PHRASES.search(candidate):
        return "multiple-choice wording"
    for ans in answer_strings(q):
        if contains_phrase(candidate, ans) and not contains_phrase(original, ans):
            return f"leaks answer ({ans})"
    return None


def split_lines(text):
    """One rewrite per line; strip numbering like '1.' or '-' the model adds anyway."""
    lines = [re.sub(r"^\s*(\d+[.)]|-)\s*", "", line).strip() for line in text.splitlines()]
    return [line for line in lines if line]


# --------------------------------------------------------------------------- #
# Main loop (model-independent, so it can be tested with a fake paraphraser)
# --------------------------------------------------------------------------- #

def load_questions():
    questions = json.loads((DATA / "source_questions.json").read_text(encoding="utf-8"))
    excluded = set((DATA / "excluded_ids.txt").read_text(encoding="utf-8").split())
    return [q for q in questions if q["id"] not in excluded]


def done_ids(path):
    """Ids already written to the output file (for resuming).

    If the previous run was stopped in the middle of writing a line, that
    unfinished line is dropped so the file stays valid.
    """
    if not path.exists():
        return set()
    good_lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            good_lines.append((json.loads(line)["id"], line))
        except (json.JSONDecodeError, KeyError):
            print("Dropping an unfinished line from the previous run")
    path.write_text("".join(line + "\n" for _, line in good_lines), encoding="utf-8")
    return {qid for qid, _ in good_lines}


def run(questions, generate, out_path, log_every=25):
    """Paraphrase every question not yet in out_path; append one line per question."""
    already = done_ids(out_path)
    todo = [q for q in questions if q["id"] not in already]
    print(f"{len(questions)} questions, {len(already)} already done, {len(todo)} to go")

    start = time.perf_counter()
    with out_path.open("a", encoding="utf-8") as f:
        for n, q in enumerate(todo, start=1):
            t0 = time.perf_counter()
            question = clean(q["question"])
            raw = generate(question)
            candidates = [
                {"text": c, "auto_reject": reject_reason(c, question, q)}
                for c in split_lines(raw)
            ]
            record = {
                "id": q["id"],
                "question": question,
                "ground_truth_answer": q["ground_truth_answer"],
                "candidates": candidates,
                "n_passed": sum(c["auto_reject"] is None for c in candidates),
                "raw_output": raw,
                "seconds": round(time.perf_counter() - t0, 2),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()  # so a crash or Ctrl+C never loses finished questions

            if n % log_every == 0 or n == len(todo):
                elapsed = time.perf_counter() - start
                eta = elapsed / n * (len(todo) - n)
                print(f"  {n}/{len(todo)} done | {elapsed / 60:.1f} min elapsed | "
                      f"~{eta / 60:.1f} min left | last: {q['id']}")

    return time.perf_counter() - start


def summarize(path):
    """Counts over the whole output file."""
    records = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    passed = [r["n_passed"] for r in records]
    return {
        "questions": len(records),
        "candidates_total": sum(len(r["candidates"]) for r in records),
        "candidates_passed": sum(passed),
        "questions_with_0_passed": sum(p == 0 for p in passed),
        "questions_with_1_passed": sum(p == 1 for p in passed),
        "questions_with_2plus_passed": sum(p >= 2 for p in passed),
        "total_generation_seconds": round(sum(r["seconds"] for r in records), 1),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None,
                        help="only process the first N questions (for a quick check)")
    args = parser.parse_args()

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    questions = load_questions()
    if args.limit:
        questions = questions[:args.limit]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16).to(device)
    model.eval()
    print("Running on", device)

    def generate(question):
        """Greedy decoding, so the same question always gives the same output."""
        messages = [{"role": "user", "content": PROMPT.format(question=question)}]
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)

    run_seconds = run(questions, generate, OUT_FILE)

    summary = summarize(OUT_FILE)
    meta = {
        "model": MODEL_NAME,
        "prompt": PROMPT,
        "generation": {"do_sample": False, "max_new_tokens": MAX_NEW_TOKENS, "dtype": "bfloat16"},
        "filters": ["not a question", "same as original", "multiple-choice wording", "leaks answer"],
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "transformers": transformers.__version__},
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "this_run_seconds": round(run_seconds, 1),
        "summary": summary,
    }
    META_FILE.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nSummary:")
    for key, value in summary.items():
        print(f"  {key:<30} {value}")
    print(f"\nWrote {OUT_FILE}\nWrote {META_FILE}")


if __name__ == "__main__":
    main()