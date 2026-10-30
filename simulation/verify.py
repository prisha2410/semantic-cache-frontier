"""Step 8a: run both verifiers on every query's nearest cached question.

A verifier only matters when the cache would serve a hit, so we check the
pairs with similarity >= the lowest threshold we sweep (0.75). Whether a
verifier accepts a pair does not depend on the threshold, so one run covers
the whole sweep.

Verifiers:
    rule   rule_guard.py: reject if negation, numbers, units or comparatives differ.
           Frozen before looking at its effect on tuning; best case for our edits.
    judge  Qwen2.5-1.5B-Instruct asked whether the two questions have the same
           answer. Greedy decoding, so the same pair always gets the same verdict.
           An answer that is neither "yes" nor "no" counts as a rejection
           (the safe choice: the query goes to the LLM).

Output:
    results/verdicts_minilm.csv, results/verdicts_mpnet.csv
        the nearest_<model>.csv columns for pairs with similarity >= 0.75, plus
        rule_accept, rule_reason, judge_raw, judge_accept
    results/judge_cache.jsonl   one line per unique (query, cached question) pair,
                                so the run can be stopped and resumed, and a pair
                                shared by both models is judged once
    results/verify_meta.json    model, prompt, settings, versions, timing

Run from the repo root:
    python simulation/verify.py --limit 20   # quick check on the first 20 pairs
    python simulation/verify.py              # all pairs
"""
import argparse
import csv
import json
import platform
import time
from pathlib import Path

from rule_guard import rule_guard

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
CACHE_FILE = RESULTS / "judge_cache.jsonl"
META_FILE = RESULTS / "verify_meta.json"

MODELS = ["minilm", "mpnet"]
MIN_THRESHOLD = 0.75

JUDGE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 3

PROMPT = """Question A: {query}
Question B: {cached}

Do these two questions have exactly the same correct answer? Answer only "yes" or "no"."""


def parse_verdict(raw):
    """'Yes.' -> True, 'no' -> False, anything else -> None."""
    word = raw.strip().lower().strip(".!,\"' ")
    if word.startswith("yes"):
        return True
    if word.startswith("no"):
        return False
    return None


def load_pairs():
    """Rows per model with similarity >= MIN_THRESHOLD, and the unique pairs among them."""
    rows = {}
    pairs = []
    seen = set()
    for model in MODELS:
        with open(RESULTS / f"nearest_{model}.csv", encoding="utf-8") as f:
            rows[model] = [r for r in csv.DictReader(f) if float(r["similarity"]) >= MIN_THRESHOLD]
        for r in rows[model]:
            key = (r["query"], r["nearest_question"])
            if key not in seen:
                seen.add(key)
                pairs.append(key)
    return rows, pairs


def load_cache():
    cache = {}
    if CACHE_FILE.exists():
        with open(CACHE_FILE, encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                cache[(item["query"], item["cached"])] = item["raw"]
    return cache


def run_judge(pairs, cache, judge):
    """Judge every pair not yet in the cache, appending each result as it is made."""
    todo = [p for p in pairs if p not in cache]
    print(f"{len(pairs)} unique pairs, {len(pairs) - len(todo)} already judged, {len(todo)} to go")
    start = time.time()
    with open(CACHE_FILE, "a", encoding="utf-8") as f:
        for i, (query, cached) in enumerate(todo, 1):
            raw = judge(query, cached)
            cache[(query, cached)] = raw
            f.write(json.dumps({"query": query, "cached": cached, "raw": raw}, ensure_ascii=False) + "\n")
            f.flush()
            if i % 100 == 0 or i == len(todo):
                elapsed = time.time() - start
                print(f"  {i}/{len(todo)}  {elapsed:.0f}s elapsed, ~{elapsed / i * (len(todo) - i):.0f}s left")
    return time.time() - start


def write_verdicts(rows, cache):
    summary = {}
    for model in MODELS:
        out = []
        for r in rows[model]:
            key = (r["query"], r["nearest_question"])
            if key not in cache:
                continue  # only possible with --limit
            accept, reason = rule_guard(*key)
            verdict = parse_verdict(cache[key])
            out.append({**r, "rule_accept": accept, "rule_reason": reason,
                        "judge_raw": cache[key], "judge_accept": verdict is True})
        if not out:
            print(f"No judged pairs for {model} yet (try a larger --limit)")
            continue
        path = RESULTS / f"verdicts_{model}.csv"
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(out[0].keys()))
            writer.writeheader()
            writer.writerows(out)
        print(f"Wrote {path} ({len(out)} rows)")

        # Share of pairs each verifier accepts, per query set
        summary[model] = {}
        for qs in sorted({r["query_set"] for r in out}):
            sub = [r for r in out if r["query_set"] == qs]
            summary[model][qs] = {
                "pairs": len(sub),
                "rule_accept": round(sum(r["rule_accept"] for r in sub) / len(sub), 3),
                "judge_accept": round(sum(r["judge_accept"] for r in sub) / len(sub), 3),
                "judge_unparsed": sum(parse_verdict(r["judge_raw"]) is None for r in sub),
            }
    return summary


def make_qwen_judge():
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(JUDGE_MODEL, dtype=torch.bfloat16).to(device)
    model.eval()
    print("Running on", device)

    def judge(query, cached):
        """Greedy decoding, so the same pair always gives the same verdict."""
        messages = [{"role": "user", "content": PROMPT.format(query=query, cached=cached)}]
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)

    versions = {"python": platform.python_version(), "torch": torch.__version__,
                "transformers": transformers.__version__}
    device_name = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
    return judge, versions, device_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="only judge the first N unique pairs")
    args = parser.parse_args()

    rows, pairs = load_pairs()
    if args.limit:
        pairs = pairs[:args.limit]

    judge, versions, device_name = make_qwen_judge()
    cache = load_cache()
    run_seconds = run_judge(pairs, cache, judge)
    summary = write_verdicts(rows, cache)

    meta = {
        "judge_model": JUDGE_MODEL,
        "prompt": PROMPT,
        "generation": {"do_sample": False, "max_new_tokens": MAX_NEW_TOKENS, "dtype": "bfloat16"},
        "unparsed_answer": "counted as reject",
        "min_similarity_checked": MIN_THRESHOLD,
        "versions": versions,
        "device": device_name,
        "this_run_seconds": round(run_seconds, 1),
        "accept_rates": summary,
    }
    META_FILE.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nShare of pairs accepted (similarity >= 0.75):")
    for model, sets in summary.items():
        print(f"  {model}")
        for qs, s in sets.items():
            print(f"    {qs:<12} n={s['pairs']:<5} rule {s['rule_accept']:.3f}   "
                  f"judge {s['judge_accept']:.3f}   unparsed {s['judge_unparsed']}")
    print(f"\nWrote {META_FILE}")


if __name__ == "__main__":
    main()
