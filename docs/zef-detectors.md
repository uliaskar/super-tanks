# ZEF detectors for tool outputs

ZEF's regex filter checks every tool output for known injection patterns.
The ZEF detectors add four checks on top, for injections the patterns miss:
reworded or translated phrases, misspelled phrases, orders hidden in data
(JSON, CSV, HTML comments) and orders without any known phrase. They run on
the tool result in the gateway, right after the regex filter, and can only
make its verdict stricter. They use no LLM and run locally.

They are **off by default**. With them on, the regex verdict stands whenever
they cannot run (missing library or file, checksum mismatch, error), and the
reason is logged.

## Install

1. Install the extra, CPU or NVIDIA GPU (one of the two):

   ```bash
   pip install -e ".[zef]"        # CPU
   pip install -e ".[zef-gpu]"    # NVIDIA GPU (needs CUDA and cuDNN for onnxruntime-gpu)
   ```

2. Download the model (about 480 MB, once):

   ```bash
   python -m scripts.zef_fetch_model
   ```

   This fetches the official ONNX export of
   `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` at a pinned
   revision into `data/zef/` and checks each file's SHA-256 against the value
   pinned in `core/security/zef_phrase_similarity.py`. A file that does not
   match is deleted.

3. Switch the detectors on:

   ```bash
   export ST_ZEF_DETECTORS=1
   ```

At the first tool output, the log says `[ZEF] detectors loaded` and, from
phrase similarity, whether it runs on `CUDAExecutionProvider` (GPU) or
`CPUExecutionProvider`. If anything is missing it says `[ZEF] detectors
unavailable, regex filter only: <reason>`.

## What a verdict means

| Verdict | When | Effect in the gateway |
|---|---|---|
| BLOCK | the regex filter blocks, or two checks flag | output withheld from the agent |
| WARN | one check flags, or two give a medium signal | flagged sentences and the sentence before and after each removed, rest passed on, marked untrusted; withheld if a sentence cannot be found |
| PASS | otherwise | output passed on |

Flags appear in `matched_patterns` as `zef:phrase_similarity`,
`zef:phrase_match`, `zef:instruction` and `zef:hidden_order` (`:medium` for a
medium signal); the segments that raised them are in `flagged_segments`, the
sentences removed on a WARN in `remove`. A WARN from the regex filter alone
keeps the output, marked untrusted.

## Files

| File | What it is | Where it comes from |
|---|---|---|
| `core/security/zef_params/zef_params.json` | phrase list, thresholds | in the repository |
| `core/security/zef_params/zef_instruction.json.gz` | instruction detector weights | in the repository |
| `data/zef/model.onnx`, `data/zef/tokenizer.json` | phrase-similarity model | `scripts/zef_fetch_model.py` |

The parameter files are produced by `research/zef-v2/harness/export_zef_params.py`,
where the detectors were evaluated ([research/zef-v2](../research/zef-v2/README.md)); their checksums are
pinned in `core/security/zef_combine.py`. A changed file is refused.

## Cost

Time per tool output, all checks, on CPU (200 sampled test outputs, 2026-11-01):
median 38 ms, 95th percentile 113 ms, maximum 398 ms. Long outputs cost the
most, since every sentence is embedded; a GPU speeds up phrase similarity.
Install size: onnxruntime and tokenizers about 56 MB, the model 470 MB.
