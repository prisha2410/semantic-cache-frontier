"""Embed every text the cache simulator needs, with both embedding models.

Texts (one row each in data/embeddings/embedding_index.csv):
    cache                787 TriviaQA questions (cleaned) + 124 constructed cached
                         questions from the adversarial set
    query_paraphrase     1,574 paraphrases (2 per TriviaQA question)
    query_adversarial    216 near-duplicates

Outputs (data/embeddings/):
    embedding_index.csv  row number -> text_id, role, category, target, answer, text
    minilm.npy, mpnet.npy  float32 matrix, one L2-normalised row per text,
                         so cosine similarity = dot product
    embed_meta.json      models, versions, revisions, shapes, timing

Run from the repo root:
    python embeddings/embed.py
"""
import csv
import json
import platform
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
OUT = DATA / "embeddings"

MODELS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
}
BATCH_SIZE = 64

INDEX_FIELDS = ["row", "text_id", "role", "category", "target_cache_id", "answer", "text"]


def build_index():
    """Collect every text with its role and the cache entry it relates to."""
    rows = []

    def add(text_id, role, category, target, answer, text):
        rows.append({"row": len(rows), "text_id": text_id, "role": role, "category": category,
                     "target_cache_id": target, "answer": answer, "text": text})

    paraphrases = json.loads((DATA / "paraphrases.json").read_text(encoding="utf-8"))
    with (DATA / "adversarial.csv").open(encoding="utf-8", newline="") as f:
        adversarial = list(csv.DictReader(f))

    # Cache: TriviaQA questions (cleaned text, same as used for paraphrasing) ...
    for q in paraphrases:
        add(q["id"], "cache", "triviaqa", q["id"], q["ground_truth_answer"], q["question"])
    # ... plus the cached side of every constructed adversarial pair.
    for a in adversarial:
        if a["source_id"] == "constructed":
            add(a["adv_id"] + "_c", "cache", "constructed", a["adv_id"] + "_c",
                a["cached_answer"], a["cached_question"])

    # Queries that should hit: paraphrases point at their own TriviaQA question.
    for q in paraphrases:
        for k, p in enumerate(q["paraphrases"], start=1):
            add(f"{q['id']}_p{k}", "query_paraphrase", p["source"], q["id"],
                q["ground_truth_answer"], p["text"])

    # Queries that should not hit: near-duplicates point at the cache entry they imitate.
    cache_ids = {r["text_id"] for r in rows if r["role"] == "cache"}
    for a in adversarial:
        target = a["source_id"] if a["source_id"] != "constructed" else a["adv_id"] + "_c"
        assert target in cache_ids, f"{a['adv_id']}: target {target} is not in the cache"
        add(a["adv_id"], "query_adversarial", a["category"], target, a["query_answer"], a["query"])

    ids = [r["text_id"] for r in rows]
    assert len(ids) == len(set(ids)), "duplicate text_id"
    return rows


def write_index(rows, path):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def embed_all(encode, texts):
    """encode(list_of_texts) -> array; normalise rows and check the result."""
    vectors = np.asarray(encode(texts), dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    assert vectors.shape[0] == len(texts)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4)
    return vectors


def model_revision(name):
    """Commit hash of the model on the Hub, so the exact weights are recorded."""
    try:
        from huggingface_hub import HfApi
        return HfApi().model_info(name).sha
    except Exception as exc:  # offline or hub error: record why instead of failing
        return f"unknown ({type(exc).__name__})"


def main():
    import sentence_transformers
    import torch
    import transformers
    from sentence_transformers import SentenceTransformer

    OUT.mkdir(parents=True, exist_ok=True)
    rows = build_index()
    write_index(rows, OUT / "embedding_index.csv")
    texts = [r["text"] for r in rows]
    roles = {}
    for r in rows:
        roles[r["role"]] = roles.get(r["role"], 0) + 1
    print(f"{len(texts)} texts: {roles}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    meta = {"texts": len(texts), "by_role": roles, "device": device,
            "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
            "versions": {"python": platform.python_version(), "torch": torch.__version__,
                         "transformers": transformers.__version__,
                         "sentence_transformers": sentence_transformers.__version__,
                         "numpy": np.__version__},
            "normalised": True, "batch_size": BATCH_SIZE, "models": {}}

    for short, name in MODELS.items():
        print(f"\nEncoding with {name} on {device} ...")
        model = SentenceTransformer(name, device=device)
        start = time.perf_counter()
        vectors = embed_all(
            lambda t: model.encode(t, batch_size=BATCH_SIZE, convert_to_numpy=True,
                                   normalize_embeddings=True, show_progress_bar=True),
            texts)
        seconds = time.perf_counter() - start
        np.save(OUT / f"{short}.npy", vectors)
        meta["models"][short] = {"name": name, "revision": model_revision(name),
                                 "shape": list(vectors.shape),
                                 "max_seq_length": model.max_seq_length,
                                 "seconds": round(seconds, 1)}
        print(f"  {vectors.shape} in {seconds:.1f} s -> {OUT / (short + '.npy')}")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    (OUT / "embed_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT / 'embedding_index.csv'}\nWrote {OUT / 'embed_meta.json'}")


if __name__ == "__main__":
    main()
