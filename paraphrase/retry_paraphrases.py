"""Second round for questions that got fewer than 2 usable paraphrases.

In the first round the model sometimes wrote only one rewrite instead of five.
Here we ask for ONE rewrite at a time, and each request lists the rewrites we
already have, so the model is pushed to say something new. Same model, same
greedy decoding and same filters as round 1, plus a duplicate check.

New candidates are added to data/paraphrase_candidates.jsonl with "round": 2.
Each retried question gets "retry_done": true, so the script can be stopped
and run again without redoing work.

Run from the repo root:
    python paraphrase/retry_paraphrases.py
"""
import json
import time

import generate_paraphrases as gp  # same folder: reuse model name, filters and helpers

MIN_PASSED = 2      # retry questions with fewer passing candidates than this
TARGET_PASSED = 4   # stop asking once a question has this many passing candidates
MAX_REQUESTS = 4    # at most this many extra requests per question

RETRY_PROMPT = """Rewrite the question below once, in different words.
Rules:
- Keep the exact meaning: the correct answer must stay the same.
- Ask for the same thing as the original (the same person, object or place being asked about).
- Keep every name, number, date and quoted phrase exactly as written.
- Do not include the answer or any hint about it.
- Do not write answer options, lists, or phrases like "which of the following" or "which of these".
- The rewrite must be a question ending with "?".
- It must be worded differently from the original and from every earlier rewrite listed below.
- Write only the rewrite, with no numbering and no extra text.

Question: {question}

Earlier rewrites:
{earlier}"""


def retry_reject_reason(candidate, record, q):
    """Round-1 filters, plus: not a repeat of an earlier candidate."""
    reason = gp.reject_reason(candidate, record["question"], q)
    if reason:
        return reason
    if gp.normalize(candidate) in {gp.normalize(c["text"]) for c in record["candidates"]}:
        return "duplicate of earlier rewrite"
    return None


def retry_one(record, q, generate):
    """Add up to MAX_REQUESTS new candidates to one record (changes it in place)."""
    for _ in range(MAX_REQUESTS):
        if record["n_passed"] >= TARGET_PASSED:
            break
        earlier = "\n".join(f"- {c['text']}" for c in record["candidates"]) or "- (none)"
        raw = generate(RETRY_PROMPT.format(question=record["question"], earlier=earlier))
        lines = gp.split_lines(raw)
        text = lines[0] if lines else ""
        reason = retry_reject_reason(text, record, q) if text else "empty output"
        record["candidates"].append({"text": text, "auto_reject": reason, "round": 2,
                                     "raw_output": raw})
        record["n_passed"] += reason is None
    record["retry_done"] = True


def write_records(records, path):
    """Rewrite the whole file via a temporary file, so a crash never leaves it half-written."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
                   encoding="utf-8")
    tmp.replace(path)


def run(records, questions_by_id, generate, path, log_every=10):
    todo = [r for r in records if r["n_passed"] < MIN_PASSED and not r.get("retry_done")]
    print(f"{len(todo)} questions to retry")
    start = time.perf_counter()
    for n, record in enumerate(todo, start=1):
        t0 = time.perf_counter()
        retry_one(record, questions_by_id[record["id"]], generate)
        record["retry_seconds"] = round(time.perf_counter() - t0, 2)
        write_records(records, path)  # save after every question
        if n % log_every == 0 or n == len(todo):
            elapsed = time.perf_counter() - start
            eta = elapsed / n * (len(todo) - n)
            print(f"  {n}/{len(todo)} done | {elapsed / 60:.1f} min elapsed | "
                  f"~{eta / 60:.1f} min left | last: {record['id']}")
    return time.perf_counter() - start


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = gp.OUT_FILE
    records = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    questions_by_id = {q["id"]: q for q in gp.load_questions()}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(gp.MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(gp.MODEL_NAME, dtype=torch.bfloat16).to(device)
    model.eval()
    print("Running on", device)

    def generate(prompt):
        """Greedy decoding, so the same prompt always gives the same output."""
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=100, do_sample=False)
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)

    run_seconds = run(records, questions_by_id, generate, path)

    # Add the retry settings and an updated summary to the meta file.
    summary = gp.summarize(path)
    meta = json.loads(gp.META_FILE.read_text(encoding="utf-8"))
    meta["retry"] = {
        "prompt": RETRY_PROMPT,
        "rule": f"questions with < {MIN_PASSED} passing candidates; one rewrite per request, "
                f"up to {MAX_REQUESTS} requests, stop at {TARGET_PASSED} passing",
        "generation": {"do_sample": False, "max_new_tokens": 100, "dtype": "bfloat16"},
        "extra_filter": "duplicate of earlier rewrite",
        "this_run_seconds": round(run_seconds, 1),
    }
    meta["summary_after_retry"] = summary
    gp.META_FILE.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nSummary after retry:")
    for key, value in summary.items():
        print(f"  {key:<30} {value}")
    print(f"\nUpdated {path}\nUpdated {gp.META_FILE}")


if __name__ == "__main__":
    main()