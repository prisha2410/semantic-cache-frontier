# Data source note

This note explains where our questions come from, how we picked them, what we checked
before using them, and how we built the paraphrases and adversarial near-duplicates used to
test the cache.

| Part | Script or file |
|---|---|
| Source questions (sections 1–5) | `data/build_source_questions.py`, `notebooks/explore_triviaqa.ipynb` |
| Paraphrases (section 6) | `paraphrase/generate_paraphrases.py`, `paraphrase/retry_paraphrases.py`, `paraphrase/build_paraphrases.py` |
| Adversarial near-duplicates (section 7) | `data/adversarial.csv`, `data/adversarial_answer_check.csv` |

## 1. Where the questions come from

We use **TriviaQA**, a set of trivia questions written by trivia enthusiasts, with answers.

- Hugging Face name: `mandarjoshi/trivia_qa`
- Config: `unfiltered.nocontext`
- Split: `train` (87,622 questions, the same number the Hugging Face page lists)

The UW TriviaQA page says the unfiltered version has 110K question–answer pairs and is
"more appropriate for IR-style QA". The `nocontext` part means we only get the question and
the answer, without the evidence documents. That fits our project: a semantic cache only
ever sees the question text, never a reading passage.

**Citation:**

Mandar Joshi, Eunsol Choi, Daniel Weld, and Luke Zettlemoyer. 2017. TriviaQA: A Large Scale
Distantly Supervised Challenge Dataset for Reading Comprehension. In *Proceedings of the 55th
Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*,
pages 1601–1611, Vancouver, Canada. Association for Computational Linguistics.
https://aclanthology.org/P17-1147/ (DOI: 10.18653/v1/P17-1147).
The preprint is arXiv:1705.03551.

## 2. Licence and copyright

Checked on 25 October 2026.

- On the Hugging Face dataset page the licence is listed as **unknown**.
- Both the UW TriviaQA page (https://nlp.cs.washington.edu/triviaqa/) and the
  "Licensing Information" section of the Hugging Face card say:
  > "The University of Washington does not own the copyright of the questions and documents
  > included in TriviaQA."

So nobody gives a clear licence for the data. Because of that, we only use it for this
course project. We don't redistribute the full dataset: our repo only holds the 800
questions we sampled, and each one points back to its original row (`row_index` and
`question_id` in `source_questions.json`).

(The CC BY 4.0 licence on the ACL Anthology page is for the paper, not the dataset.)

## 3. How we filtered the questions

We wanted questions that are short, have one clear answer, and contain something we can
change later to build adversarial near-duplicates (a year, a number, a unit, a comparative
like "largest", or a named entity).

The script checks each question against these rules **in order**. A question is counted
under the first rule it fails.

| Rule | What it removes | Removed |
|---|---|---:|
| Answer missing | rows with an empty answer | 0 |
| Answer too long | answers longer than 5 words | 2,626 |
| Question length | questions shorter than 4 or longer than 20 words | 11,198 |
| Time-sensitive | words like *current, recent, today, to date, reigning, as of* | 612 |
| Multi-answer | phrases like *name the / which two / what are the / list* | 996 |
| No modifiable feature | nothing we could change (no year, number, Roman numeral, comparative, unit, yes/no start or named entity) | 9,399 |
| Duplicate | same question after lowercasing and removing punctuation | 166 |

Then we took a random sample of **800** with seed **42**.

### The funnel

| Stage | Questions left |
|---|---:|
| All rows | 87,622 |
| After answer missing | 87,622 |
| After answer too long | 84,996 |
| After question length | 73,798 |
| After time-sensitive | 73,186 |
| After multi-answer | 72,190 |
| After no modifiable feature | 62,791 |
| After duplicates | 62,625 |
| **Sampled** | **800** |

### What the 800 look like

- Features found: entity 765, year 188, comparative 125, number 75, unit 10,
  Roman numeral 8. (One question can have several.)
- Answer types: WikipediaEntity 722, FreeForm 52, Numerical 24, Range 2.

## 4. Checking the answers

The filters only look at the words in a question, not at whether the answer is right.
So we checked 30 random answers (`random.Random(7)`) against outside sources. The results
are in `data/answer_spotcheck_filled.csv`.

| Verdict | Count |
|---|---:|
| OK | 25 |
| Wrong | 0 |
| Ambiguous | 5 |

The 5 ambiguous ones:

| ID | Question (short) | Problem |
|---|---|---|
| q0332 | Church that keeps the Sabbath on Saturday | More than one church fits, so there are several right answers |
| q0155 | Element besides uranium from pitchblende | Radium and polonium are both right |
| q0050 | Aircraft with nicknames like "ScareBus" | We couldn't confirm the full nickname list in a reliable source |
| q0429 | "Of which country **is** Helen Clark Prime Minister?" | She was PM from 1999 to 2008, so the present tense is wrong today |
| q0643 | Name of the Jewish people of Ethiopia | The answer "Falashas" is an outdated term many consider offensive; the accepted name is Beta Israel |

## 5. What we excluded

We removed **13** of the 800 questions, which leaves **787**. Their IDs are in
`data/excluded_ids.txt`.

**5 from the answer spot-check** (section 4): q0332, q0155, q0050, q0429, q0643.

**8 found while reviewing paraphrases** (section 6). Reading every question closely to
judge its rewrites showed problems the filters had missed:

| ID | Question (short) | Problem |
|---|---|---|
| q0043 | British Prime Minister between 1880 and 1885 | Dataset answer is wrong: it says Disraeli, but Gladstone was PM then |
| q0153 | Activity that uses Naismith's Rule | Dataset answer is cut off: "Hikers or Mountaineers or" |
| q0158 | "Who won the Third-Place play-off?" | No tournament named, so it can't be answered |
| q0204 | Mark Twain quote about a professional person | The quote is missing from the question text |
| q0272 | Linus Pauling's helical structure | Answer doubtful: Pauling's helix was the alpha helix of proteins, not DNA |
| q0502 | Ed Miliband's shadow cabinet post | Dataset answer is cut off: "ENERGY &" |
| q0740 | "WEAK VITAL OIL (anagram film)" | A puzzle, not a question |
| q0750 | "Mould, Board, Disc and Rotary" | A fragment with no question |

## 6. Paraphrases

A semantic cache should return the stored answer when someone asks the **same** question in
other words. To measure that, we need two genuine rewordings of each of the 787 questions
(1,574 paraphrases in total, in `data/paraphrases.json`).

### How we made them

1. **Generation.** We asked **Qwen2.5-1.5B-Instruct** (Apache-2.0) for 5 rewrites of each
   question, with greedy decoding (the same question always gives the same output). The
   prompt asks it to keep the meaning, keep every name, number and date, not hint at the
   answer, and not write multiple-choice wording. The exact prompt and settings are in
   `data/paraphrase_candidates_meta.json`.
2. **Automatic filters.** Each candidate is marked as rejected if it is not a question, is
   the same as the original, uses multiple-choice wording ("which of the following"), or
   contains the answer or one of its aliases.
3. **Retry.** The model often wrote only one rewrite. For questions with fewer than 2
   candidates passing, we asked again for one rewrite at a time (up to 4 requests), also
   rejecting repeats of earlier rewrites.

| | First run | After retry |
|---|---:|---:|
| Candidates | 3,613 | 4,056 |
| Passed the filters | 3,329 | 3,644 |
| Questions with 0 passing | 16 | 10 |
| Questions with 1 passing | 117 | 8 |

(These counts are for the 795 questions left after the spot-check.)

4. **Human review.** Passing the filters does not mean a rewrite asks the same thing, so we
   read the candidates ourselves (`data/paraphrase_review.csv`). Each was accepted or
   rejected with a reason. The most common rejection reasons were "garbled" (95),
   "different question" (77), "same" as the original (76), and "vague" (41). Once a question
   had two accepted rewrites we stopped reading its remaining candidates.

| Review result | Candidates |
|---|---:|
| Accepted | 1,281 |
| Rejected by us | 1,027 |
| Rejected by the automatic filters | 412 |
| Not read (question already had two, or was flagged) | 1,336 |

5. **Manual paraphrases.** Where fewer than two model rewrites were good, we wrote the
   missing ones by hand (`data/paraphrases_manual.tsv`).

**Final set:** 787 questions × 2 paraphrases = 1,574. Of these, **1,278** come from the
model and **296** were written by hand (108 questions have one manual paraphrase, 94 have
two).

### Checking our own review

Because one of us did most of the review, the other did a **blind check** on 50 items
(30 we had accepted, 20 we had rejected), judging each as good or bad without seeing our
verdict (`data/review_check_partner.csv`, key in `data/review_check_key.csv`).

- Agreement: **47 of 50** (94%), Cohen's κ = **0.87**.
- All **30 accepted** paraphrases were judged good.
- The 3 disagreements were all paraphrases we had rejected that the check judged good, so
  our review was, if anything, too strict.

## 7. Adversarial near-duplicates

These are queries that look almost the same as a cached question but have a **different
answer**. If the cache serves the stored answer, that is an error. There are 216 in
`data/adversarial.csv`, in five categories:

| Category | Example change | Items | From our TriviaQA questions | Written by us |
|---|---|---:|---:|---:|
| Number | 1987 → 1988 | 36 | 36 | 0 |
| Entity | 'To Catch a Thief' → 'Rebel Without a Cause' | 50 | 50 | 0 |
| Comparative | largest → smallest | 50 | 6 | 44 |
| Unit | kilometres → miles | 30 | 0 | 30 |
| Negation | "Which of A or B is X?" → "...is not X?" | 50 | 0 | 50 |

- **"From our TriviaQA questions"** means the cached question is one of the 787, and the
  query is our edited version of it.
- **"Written by us"** means we wrote both the cached question and the query, because too
  few TriviaQA questions had the feature (only 10 of the 800 contain a unit, for example).
  These 124 cached questions are added to the cache, but only when testing their own
  category, so the categories don't interfere with each other.
- Each row records the change, the query's answer and an evidence link (or the calculation,
  for computed answers like "How many hours are in a leap year?").

### Checking the answers

We checked **40 items** (8 per category) against their evidence
(`data/adversarial_answer_check.csv`). All 40 were **correct**.

A few evidence links could not be opened when we checked (the archived NIST unit-conversion
PDF, the George H. W. Bush 1992 campaign page, and some WorldAtlas URLs). In those cases we
confirmed the fact from another source and say which one in the notes column.

## 8. Limitations

- **The filters are simple word-matching rules, so they miss things.** For example, q0429
  (Helen Clark) got past the time-sensitive filter because it has none of the trigger words,
  and the paraphrase review later found 8 more flawed questions (section 5).
- **The spot-check is small.** 30 questions is under 4% of the 800. Finding 0 wrong answers
  doesn't mean there are none. By the "rule of three", 0 errors in 30 still allows a true
  error rate of up to about 10% (3/30).
- **More ambiguous or outdated questions are probably still in the set.** For example, q0004
  gives Sudan as Africa's largest country, which stopped being true when South Sudan became
  independent in 2011. This only matters for paraphrase queries, where the cached answer is
  scored against TriviaQA's own answer, so it does not create cache errors in our results.
- **One person reviewed most paraphrases.** The blind check (κ = 0.87 on 50 items) suggests
  the review is reliable, but 50 items is a small sample.
- **The paraphrases come from one small model plus our own writing.** Real users may reword
  questions in ways neither produces.
- **Most comparative, unit and negation items are written by us**, not taken from real
  questions, so they may be cleaner and more uniform than real traffic.
- **The adversarial sets are small** (30–50 per category), so error rates have wide
  confidence intervals. We report 95% Wilson intervals with every rate.
- **Both embedding models were trained on data that includes TriviaQA** (their model cards
  list it in the training mix), so they may have seen these exact questions. This could make
  paraphrase matching look easier than it would be on new questions.
