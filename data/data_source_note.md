# Data source note

This note explains where our questions come from, how we picked them, and what we checked
before using them. Everything here can be reproduced with `data/build_source_questions.py`
and `notebooks/explore_triviaqa.ipynb`.

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

We removed these 5 questions, which leaves **795**. Their IDs are in `data/excluded_ids.txt`:
q0332, q0155, q0050, q0429, q0643.

## 6. Limitations

- **The filters are simple word-matching rules, so they miss things.** For example, q0429
  (Helen Clark) got past the time-sensitive filter because it has none of the trigger words.
- **The spot-check is small.** 30 questions is under 4% of the 800. Finding 0 wrong answers
  doesn't mean there are none. By the "rule of three", 0 errors in 30 still allows a true
  error rate of up to about 10% (3/30).
- **More ambiguous questions are probably still in the set.** 5 out of 30 (about 1 in 6)
  were ambiguous, and we only removed the ones we actually checked. We'll keep this in mind
  when we read the cache error rates later.
- **Few questions have units.** Only 10 of the 800 contain a unit, so the "changed unit"
  adversarial category won't have many natural starting questions.