"""Quick test: paraphrase 10 source questions and print the candidates.

Nothing is saved. We only look at the output to judge quality before
running all 795 questions.

Run from the repo root:
    python paraphrase/try_paraphraser.py
"""
import json
import random
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

REPO = Path(__file__).resolve().parent.parent
MODEL_NAME = "humarin/chatgpt_paraphraser_on_T5_base"
N_TEST = 10
N_CANDIDATES = 5

# 1. Load the questions and drop the 5 excluded ids.
questions = json.loads((REPO / "data" / "source_questions.json").read_text(encoding="utf-8"))
excluded = set((REPO / "data" / "excluded_ids.txt").read_text(encoding="utf-8").split())
questions = [q for q in questions if q["id"] not in excluded]
print(len(questions), "questions after exclusions")

# 2. Pick 10 at random (fixed seed, so the same 10 every time).
sample = random.Random(0).sample(questions, N_TEST)

# 3. Load the model on the GPU (falls back to CPU if there is none).
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
model.eval()
print("Running on", device)


def paraphrase(question):
    """Return N_CANDIDATES paraphrases using normal (deterministic) beam search.

    Settings follow the model card, minus the group-beam-search options
    (num_beam_groups, diversity_penalty) and temperature, which normal beam
    search does not use.
    """
    inputs = tokenizer(f"paraphrase: {question}", return_tensors="pt",
                       max_length=128, truncation=True).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            num_beams=5,
            num_return_sequences=N_CANDIDATES,
            repetition_penalty=10.0,
            no_repeat_ngram_size=2,
            max_length=128,
        )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


# 4. Print each question with its candidates.
for q in sample:
    print("\n" + "=" * 80)
    print(f"{q['id']}  ORIGINAL: {q['question']}   (answer: {q['ground_truth_answer']})")
    for i, cand in enumerate(paraphrase(q["question"]), start=1):
        print(f"   {i}. {cand}")