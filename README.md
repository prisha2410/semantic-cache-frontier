# Mapping the cost–correctness frontier of semantic caching

A semantic cache answers a new question with a stored answer when the question is
*similar enough* to one it has seen before: embed the query, find the nearest cached
question, and serve its answer if the cosine similarity is above a threshold. Every hit
saves an LLM call, but a question that only *looks* the same ("Which of A or B **is not**
X?") gets the wrong answer.

We measure that trade-off: how many calls a semantic cache saves on genuine rephrasings,
and how many wrong answers it serves on near-duplicates, as the threshold changes, and
whether a verifier makes a safe setting possible.

## The one command

```powershell
python reproduce.py
```

This regenerates every result and figure from the committed data in under a minute on a
laptop, **with no GPU** (the LLM judge's verdicts are reused from `results/judge_cache.jsonl`),
then checks that the numbers match the committed ones and prints
`All main results reproduced.`

Setup (Windows PowerShell, Python 3.12):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Experimental setup

| | |
|---|---|
| **Cache** | 787 TriviaQA questions with answers (`data/source_questions.json`, minus `data/excluded_ids.txt`) |
| **Genuine rephrasings** | 1,574 paraphrases, 2 per cached question, reviewed by hand (`data/paraphrases.json`) |
| **Near-duplicates** | 216 adversarial queries with a different answer: number 36, entity 50, comparative 50, unit 30, negation 50 (`data/adversarial.csv`) |
| **Varied** | similarity threshold (0.75–0.98, steps of 0.01); embedding model (all-MiniLM-L6-v2, all-mpnet-base-v2); verifier (none, rule guard, LLM judge, both) |
| **Held fixed** | cache contents, query sets, answer scorer, normalised embeddings, greedy decoding for every LLM step; each adversarial category is tested against its own cache |
| **Baselines** | exact-match cache; TF-IDF word-overlap cache (thresholds 0.30–0.98) |
| **Metrics** | paraphrase hit rate (savings); error rate per category = share of near-duplicates served a wrong answer; per 1,000 queries: calls saved and wrong answers |
| **Error bars** | 95% Wilson intervals on every rate; bootstrap (1,000 resamples) for the safe threshold |
| **Checks against hand labels** | answer scorer and LLM judge vs 57 hand-labelled pairs; paraphrase review vs a blind second reviewer (50 items) |

**Safe threshold:** the threshold that saves the most calls while serving at most
**1 wrong answer per 1,000 queries**, assuming 300 genuine rephrasings and D = 10, 50 or 100
near-duplicates per 1,000 (illustrative traffic, categories weighted equally).

## Key findings

1. **No threshold alone is safe.** Even at 0.98, MiniLM serves the wrong answer to 98% of
   negation queries (MPNet 92%), while saving only about 9% of paraphrases. "X" and
   "not X" sit almost on top of each other in embedding space.
2. **The stronger model is not safer.** MPNet makes more errors than MiniLM on comparative
   changes at the same paraphrase hit rate, and on unit changes over most of the curve.
3. **Embeddings beat word overlap on the main trade-off**, but TF-IDF notices "not": its
   negation error falls to 0 at threshold 0.96, where the embeddings still fail. Its safe
   point saves only 13–17 calls per 1,000.
4. **The LLM judge alone is not enough.** Qwen2.5-1.5B accepts 84% of negation pairs and
   rejects 13% of genuine paraphrases (agreement with the scorer κ = 0.45).
5. **Rule guard + LLM judge is safe at every D.** At threshold 0.75 it saves about
   **221 calls per 1,000** (bootstrap 95% range 211–229) with 0.04–0.40 wrong answers per
   1,000. The rule guard alone saves more (up to 249) but its safe threshold moves with D
   and it does not catch entity swaps.
6. **Our sample cannot certify the budget.** With 0 errors in 30–50 items per category, the
   upper ends of the 95% intervals still allow more than 1 wrong answer per 1,000 for every
   setting, so the safe thresholds are point estimates.
7. **The answer scorer agrees with hand labels on 57/57 pairs** (κ = 1.0), so the error
   counts can be trusted.

| Setup (D = 10) | Safe threshold | Calls saved per 1,000 | Wrong per 1,000 (upper bound) |
|---|---|---:|---|
| Either embedding model, no verifier | none | 0 | – |
| Either model + LLM judge | none | 0 | – |
| MiniLM + rule guard | 0.75 | 246 | 0.44 (2.14) |
| MiniLM + rule guard + judge | 0.75 | 221 | 0.04 (1.65) |
| TF-IDF baseline | 0.94 | 17 | 0.78 (2.61) |
| Exact-match baseline | – | 0 | 0 |

Full tables: `results/safe_thresholds.csv`, `results/bootstrap_safe_thresholds.csv`.

### Figures

| | |
|---|---|
| ![frontier](results/figures/frontier.png) | **Frontier per category:** savings vs errors, one line per model and the TF-IDF baseline |
| ![sweep](results/figures/threshold_sweep.png) | **Threshold sweep:** paraphrase hit rate and each error rate, with 95% Wilson bands |
| ![verifiers](results/figures/verifier_frontier.png) | **Verifiers:** the frontier with each verifier, against the error budget |

## Repository map

| Folder | What is in it |
|---|---|
| `data/` | source questions, paraphrases, adversarial set, hand labels, embeddings; how they were built is in [`data/data_source_note.md`](data/data_source_note.md) |
| `paraphrase/` | paraphrase generation with Qwen2.5-1.5B-Instruct (GPU) and the build step |
| `embeddings/` | `embed.py`: encodes all 2,701 texts with both models (GPU) |
| `simulation/` | `simulate.py` (sweep), `baseline.py`, `plots.py`, `rule_guard.py` + `verify.py` (verifiers), `costs.py` (safe thresholds), `bootstrap.py`, `hand_labels.py` |
| `results/` | every CSV and figure produced by `reproduce.py` |

### Rebuilding from scratch (needs a GPU)

`reproduce.py` starts from the committed paraphrases, embeddings and judge verdicts. To
rebuild those too, run in order: `data/build_source_questions.py`,
`paraphrase/generate_paraphrases.py`, `paraphrase/retry_paraphrases.py`, the hand review,
`paraphrase/build_paraphrases.py`, `embeddings/embed.py`, then delete
`results/judge_cache.jsonl` and run `reproduce.py` (the judge then runs on the GPU,
about 2.5 minutes on an RTX 3050).

## Limitations

- The rule guard targets exactly the edits we used to build the near-duplicates, so its
  results are a best case.
- The traffic mix (300 rephrasings and D near-duplicates per 1,000) is an assumption, not
  measured.
- 124 of the 216 near-duplicates (comparative, unit and negation) were written by us, and the
  adversarial sets are small (30–50 per category).
- We swept thresholds from 0.75 up; several safe thresholds sit at that edge, so lower
  thresholds might save more.
- Both embedding models were trained on data that includes TriviaQA, which may make
  paraphrase matching look easier than on new questions.
- More in [`data/data_source_note.md`](data/data_source_note.md), section 8.

## Data and models

- TriviaQA: Joshi et al., ACL 2017 (`mandarjoshi/trivia_qa`, `unfiltered.nocontext`); licence
  listed as unknown, used for this course project only.
- sentence-transformers/all-MiniLM-L6-v2 and all-mpnet-base-v2 (Apache-2.0).
- Qwen/Qwen2.5-1.5B-Instruct (Apache-2.0), for paraphrases and as the LLM judge.
