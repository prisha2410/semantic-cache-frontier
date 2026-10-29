"""Quick test: paraphrase the same 10 questions with a small instruction model,
then apply automatic filters.

The result is printed and also written to paraphrase/qwen_test_output.txt
(UTF-8), so it can be copied without terminal line-wrapping problems.

Run from the repo root:
    python paraphrase/try_qwen.py
"""
import json
import random
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
N_TEST = 10

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


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # 1. Same 10 questions as before (same exclusions, same seed).
    questions = json.loads((REPO / "data" / "source_questions.json").read_text(encoding="utf-8"))
    excluded = set((REPO / "data" / "excluded_ids.txt").read_text(encoding="utf-8").split())
    questions = [q for q in questions if q["id"] not in excluded]
    sample = random.Random(0).sample(questions, N_TEST)

    # 2. Load the model in bfloat16 (the format the weights are stored in).
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16).to(device)
    model.eval()
    print("Running on", device)

    def paraphrase(question):
        """Ask for 5 rewrites; greedy decoding, so the output is the same every run."""
        messages = [{"role": "user", "content": PROMPT.format(question=question)}]
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=300, do_sample=False)
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        lines = [re.sub(r"^\s*(\d+[.)]|-)\s*", "", line).strip() for line in text.splitlines()]
        return [line for line in lines if line]

    # 3. Generate, filter, and report kept and rejected candidates.
    report = []
    n_kept_total = 0
    for q in sample:
        question = clean(q["question"])
        block = ["=" * 80, f"{q['id']}  ORIGINAL: {question}   (answer: {q['ground_truth_answer']})"]
        kept = 0
        for i, cand in enumerate(paraphrase(question), start=1):
            reason = reject_reason(cand, question, q)
            if reason:
                block.append(f"   {i}. [REJECTED: {reason}] {cand}")
            else:
                block.append(f"   {i}. {cand}")
                kept += 1
        block.append(f"   -> {kept} passed the filters")
        n_kept_total += kept
        print("\n" + "\n".join(block))
        report += block

    report.append(f"\nTotal passed: {n_kept_total}")
    out_file = REPO / "paraphrase" / "qwen_test_output.txt"
    out_file.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"\nTotal passed: {n_kept_total}\nWrote {out_file}")


if __name__ == "__main__":
    main()