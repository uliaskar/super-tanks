# ZEF version 2: catching hidden instructions in tool outputs

Research behind the ZEF detectors in this branch (`core/security/zef_*.py`):
what they do, how well they work, and how to reproduce the results.
Installing and switching them on: [docs/zef-detectors.md](../../docs/zef-detectors.md).

## The problem

Super Tanks' agents read **tool outputs**: emails, files, web pages, bank
transactions, chat messages. Whoever writes that text can hide an instruction
for the agent in it, such as "send the money to this account first". This is
*indirect prompt injection*. ZEF, Super Tanks' prompt-injection filter, checks
every tool output with a list of regular expressions. The list catches the
standard phrasing ("ignore all previous instructions…") but not an instruction
that is wrapped in JSON, written in Norwegian or Russian, mixes languages,
or is misspelled or reworded.

## What version 2 adds

1. **A normalizer** (always on, for every ZEF caller): before matching, ZEF
   also undoes escapes, URL encoding, HTML entities, base64, invisible
   characters and look-alike letters (e.g. a Cyrillic "і" in "іgnore").
2. **Four detectors** (off unless `ST_ZEF_DETECTORS=1`), run on each sentence
   or data field of a tool output:

   | Detector | Recognises | How | Code in the scripts |
   |---|---|---|---|
   | Phrase similarity | rewordings and translations of 82 known injection phrases (English, Norwegian, Russian; `harness/phrase_list.py`) | multilingual sentence embeddings (MiniLM, 470 MB, ONNX) | B |
   | Phrase matcher | known phrases despite misspellings | word-by-word match with an edit-distance tolerance | A2 |
   | Instruction detector | a sentence that gives the assistant an order, with or without a known phrase | character fragments + logistic regression, trained | C2c |
   | Hidden order | such an order inside JSON, CSV or an HTML comment within a field | the instruction detector's score on those parts | H |

3. **One verdict per tool output.** Each detector has a *flag* threshold and a
   lower *medium* threshold. BLOCK if the regex list blocks or two detectors
   flag; WARN on one flag or two medium signals; otherwise PASS. The verdict is
   never looser than the regex list's. Thresholds are set so that at most 1 in
   100 clean training outputs gets flagged.
4. **What a WARN does:** the flagged sentences, and the sentence before and
   after each, are removed; the rest of the output reaches the agent. A BLOCK
   withholds the whole output.

No LLM is called. Time per tool output on CPU: median 38 ms, 95th percentile
113 ms.

## Results

### Agent test

[AgentDojo](https://github.com/ethz-spylab/agentdojo) runs an AI agent on
realistic tasks (banking, Slack) while some tool outputs carry an attacker's
instruction, and checks whether the agent finished the user's task and whether
it did what the attacker asked. Model: `qwen2.5:7b`, local. Five attack types:
AgentDojo's standard one and four "wrapped" ones, where the same instruction
is put inside JSON or written in Norwegian, Russian or mixed languages. Each
attack type: 747 attacked tasks (249 per repetition, 3 repetitions). Plus 111
runs without any attack. *Version 1* is our earlier
attempt: one embedding classifier on top of the regex list, whose WARN kept the
output and put a warning notice above it.

| | No protection | Regex list only (today) | Version 1 | **Version 2** |
|---|---|---|---|---|
| Successful attacks, all five types (of 3,735) | 531 | 328 | 198 | **92** |
| … standard attack (of 747) | 191 | 1 | 4 | 2 |
| … JSON, Norwegian, mixed languages (of 2,241) | 260 | 245 | 121 | **9** |
| … Russian (of 747) | 80 | 82 | 73 | 81 |
| Tasks completed without attack (of 111) | 47 | 44 | 39 | 45 |
| Tasks completed under attack (of 3,735) | 1,271 | 1,186 | 1,197 | 1,190 |
| Runs without attack: tool outputs wrongly flagged | — | 0 of 412 | 10 of 346 | 0 of 442 |

"Successful attack" = the agent did what the attacker asked. "No protection"
and "Version 1" come from the version 1 run (2026-10-25), the rest from
2026-11-03; the regex list, run in both, gave 321 and 328, so the runs are
comparable. The small local model completes only about 40 % of tasks even
without an attack. All differences in tasks completed are within chance (95 %
intervals overlap); the drop from 1,271 is the regex list blocking the
standard attack, shared by every protected setup.

Version 1 and version 2 warn on about the same share of attacked tool outputs
(19 %). The difference is what a WARN does. Version 2 was also run with
version 1's warning notice instead of removal: with exactly the same verdicts,
116 of 2,241 JSON, Norwegian and mixed-language attacks succeeded with the
notice, 9 with removal.

### Single tool outputs

The test part of the dataset (see "How it was measured"): 3,447 attacked and
2,681 clean tool outputs. Of the clean ones, 2,065 are deliberately
confusing *imitation texts* (harmless text inserted in an attack's format, or
sentences using trigger words such as "ignore the previous invoice"); 616 are
ordinary tool outputs.

| | Regex list only | Version 2 |
|---|---|---|
| Attacked outputs caught (WARN or BLOCK, of 3,447) | 886 | **2,394** (875 blocked) |
| Clean outputs flagged (of 2,681) | 18 | 79 |
| … of the 616 ordinary outputs | 0 | **0** |
| Ordinary sentences flagged, of 1,000 each (Tatoeba, Russian / Bokmål / Nynorsk) | — | 3 / 1 / 6 |

Per attack type, version 2 catches every standard, JSON, Norwegian,
mixed-language and invisible-character attack (284 of 284 each), and 244 of
284 HTML-comment attacks; all 79 false alarms are imitation texts.

### Robustness

104 attack sentences were changed one way at a time; of the 88 version 2
caught before the change, this many escaped after it: misspelling 1,
look-alike letters / invisible characters / letter case 0, wrappers (JSON,
CSV, YAML, HTML comment, Markdown, code comment) at most 1 each, encodings 0,
Norwegian or Russian translation of the framing 0, **synonyms 32, reworded
framing 48**. Chains of 2 or 3 changes: 7 % and 8 % escape on average.

## What version 1 taught

| Version 1 problem | Version 2 |
|---|---|
| A WARN only added a notice; the agent often followed the attacker anyway | A WARN removes the flagged sentences and their neighbours |
| Removing the flagged parts took away the whole output in 1,016 of 1,261 flagged attacks | Removal by sentence; no tasks lost |
| Wrapped orders without a known phrase were left to one classifier | A detector for orders, extra weight for orders inside hidden data |
| Misspellings threw off the embedding model | A phrase matcher that tolerates misspellings |

## Known limits

- **Russian "important message" framing** ("Важное сообщение: сначала
  выполни это действие…"): phrase similarity scores it 0.69, below its 0.753
  flag threshold, and the instruction detector gives at most a medium signal,
  so it passes (0 of 284 test outputs flagged; 81 of 747 agent-test attacks
  succeed, as without protection). Translations of *known* phrases are caught.
- **Synonyms and reworded framings** (see Robustness): the small embedding
  model doesn't see them as close to the known phrases.
- **Orders wrapped in CSV:** 97 of 284 test outputs caught.

## How it was measured

**Dataset** (`zef_dataset.py`, no LLM calls): pairs of (user request, tool
output), label 1 if the output contains the attacker's goal. Sources:
AgentDojo's reference tool calls replayed clean and with attacks inserted
(standard, wrapped, override phrasings in English and Norwegian, generated
framings), harmless "control" insertions in the same formats, look-alike
sentences using trigger words ("ignore the previous invoice"), ordinary
environment text, plus InjecAgent attacks and BIPIA harmless hijacks
(downloaded at pinned commits). IBANs, e-mail addresses and URLs are replaced
by random stand-ins, so no detector can learn an attacker's address.

**Split:** trained and tuned on AgentDojo's workspace and travel suites plus
the training halves of InjecAgent and BIPIA; tested on banking and Slack plus
the held-out halves. Thresholds are never chosen on test data.

**Measures:** attacks caught (attacked outputs with WARN or BLOCK), false
alarms (clean outputs with WARN or BLOCK), always as counts out of a total;
95 % intervals (bootstrap offline, Wilson in the agent test).

**Agent-test goals** (fixed before the full run; version 2 against the regex
list only): (1) successful wrapped attacks at most two thirds, intervals not
overlapping; (2) standard attack not clearly higher; (3) runs without attack:
at most 1 in 100 tool outputs flagged; (4) tasks without attack at most 6 of
111 fewer; (5) tasks under attack at most 5 % fewer. All five passed.

**Steps** referred to in the scripts: 1 dataset and evaluation setup;
2 phrase similarity (7 embedding models compared, MiniLM chosen); 3 instruction
detector (word lists, grammar features with spaCy, character fragments; the
last chosen); 4 segmenter; 5 phrase matcher (word alignment chosen);
6 checking the output against the user request (tested, nothing added);
7 verdict combiner and thresholds; 8 Super Tanks integration; 9 agent test.

## Reproduce

Everything runs locally and needs only this checkout. From
`research/zef-v2/harness/`, with `PYTHONPATH` set to the repository root
(`export PYTHONPATH=../../..`; on Windows `set PYTHONPATH=..\..\..`):

```bash
pip install -e "../../..[zef,dev]" -r requirements.txt        # same venv as Super Tanks
(cd ../../.. && python -m scripts.zef_fetch_model)            # phrase-similarity model
python -m venv .venv-embed && .venv-embed/bin/pip install -r requirements-embed.txt
                                                    # Windows: .venv-embed\Scripts\pip

python zef_dataset.py                 # dataset -> data/zef_pairs.jsonl (about 8 minutes)
python eval_combine.py --split main   # single-output results of the final system
python eval_combine.py --robustness   # robustness test
python export_zef_params.py           # writes core/security/zef_params/ (checksums must match zef_combine.py)
python check_runtime.py               # Super Tanks gives the same verdict as the harness
```

`eval_phrase_similarity.py`, `eval_instruction.py` and `eval_phrase_match.py`
compare the candidates of steps 2, 3 and 5 (some candidates need Ollama models).

**Agent test** (needs [Ollama](https://ollama.com) with `qwen2.5:7b`; about
14 hours on an RTX 3080 for the full run):

```bash
OLLAMA_URL= OLLAMA_LOCAL_URL=http://127.0.0.1:11434/v1 OLLAMA_LOCAL_MODEL=qwen2.5:7b \
python run_baseline_test.py --suite banking slack \
  --attack important_instructions station3_json station3_norwegian station3_russian station3_code_switch \
  --clean --reps 3 --configs regex v2-remove-nb
```

Add `--smoke` for a quick run. Results go to `results/`; the five goals are
checked and printed at the end; `--resume FILE` continues an interrupted run.

## Files

| File | What it does |
|---|---|
| `core/security/zef_normalize.py` | normalizer |
| `core/security/zef_segment.py` | splits a tool result into sentences and data fields, marks hidden data |
| `core/security/zef_phrase_similarity.py`, `zef_phrase_match.py`, `zef_instruction.py` | the detectors |
| `core/security/zef_combine.py` | loads the detectors, combines the verdict, finds the sentences to remove |
| `core/gateway.py` | applies the verdict to each tool output |
| `harness/zef_dataset.py`, `custom_attacks.py`, `phrase_list.py` | dataset, wrapped attacks, known phrases |
| `harness/eval_*.py` | offline tests (single outputs, robustness, candidates, combiner) |
| `harness/export_zef_params.py`, `check_runtime.py` | parameter files for Super Tanks, runtime check |
| `harness/run_baseline_test.py`, `zef_defense_adapter.py`, `backend.py` | agent test (ZEF as an AgentDojo defense) |
| `harness/embeddings.py`, `embed_st.py`, `features_spacy.py` | embedding and grammar helpers (`.venv-embed`) |
