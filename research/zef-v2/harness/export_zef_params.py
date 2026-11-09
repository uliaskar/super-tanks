"""
export_zef_params.py
====================
Step 8 (../README.md): writes the parameter files the Super Tanks runtime
loads, from exactly the system eval_combine.py measured (step 7):

  zef_params.json         phrase list, phrase-matcher setting, thresholds
  zef_instruction.json.gz instruction detector (C2c): the character-fragment
                          vocabulary with each fragment's idf and weight

Both are written deterministically (sorted keys, gzip without a timestamp),
so the same data gives the same SHA-256; the script prints the checksums to
pin in core/security/zef_combine.py.

Usage:
  python export_zef_params.py [--out ../../super-tanks/core/security/zef_params]
"""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

import eval_combine as ec
import eval_phrase_match as pm
import phrase_list

DEFAULT_OUT = Path(__file__).resolve().parents[3] / "core" / "security" / "zef_params"


def _finite(x):
    """JSON has no infinity: a threshold that never fires is written as null."""
    return None if np.isinf(x) else float(x)


def export(out):
    _, _, _, _, comb = ec.fit()
    tfidf, clf = comb.c2c.tfidf, comb.c2c.model
    if (tfidf.analyzer, tfidf.ngram_range, tfidf.sublinear_tf, tfidf.norm) != ("char_wb", (2, 5), True, "l2"):
        raise SystemExit("instruction detector settings differ from what the runtime implements")
    vocab = sorted(tfidf.vocabulary_.items(), key=lambda kv: kv[1])
    coef = clf.coef_[0]
    instruction = {"ngram_range": [2, 5], "intercept": float(clf.intercept_[0]),
                   "features": [[g, float(tfidf.idf_[i]), float(coef[i])] for g, i in vocab]}
    params = {
        "phrases": phrase_list.expanded(),
        "phrase_match_min_words": pm.MIN_WORDS,
        "thresholds": {k: {"flag": _finite(h), "medium": _finite(lo)}
                       for k, (h, lo) in comb.thresholds.items()},
        "detectors": {"B": "phrase similarity", "A2": "phrase matcher", "C2c": "instruction detector",
                      "H": "an order inside hidden data (instruction detector on hidden-data segments)"},
    }
    out.mkdir(parents=True, exist_ok=True)
    files = {
        "zef_params.json": json.dumps(params, ensure_ascii=False, indent=1, sort_keys=True).encode("utf-8"),
        "zef_instruction.json.gz": gzip.compress(
            json.dumps(instruction, ensure_ascii=False, separators=(",", ":")).encode("utf-8"), mtime=0),
    }
    for name, data in files.items():
        (out / name).write_bytes(data)
        print(f"{name}: {len(data) / 1e6:.2f} MB, sha256 {hashlib.sha256(data).hexdigest()}")


def main(argv=None):
    p = argparse.ArgumentParser(description="Export the ZEF runtime parameter files (step 8).")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    export(p.parse_args(argv).out)


if __name__ == "__main__":
    main()
