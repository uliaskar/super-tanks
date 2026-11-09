"""
eval_false_alarms.py
====================
False-alarm test on innocent text that the main dataset lacks (../README.md):
how often does ZEF flag (WARN or BLOCK) text that contains no attack, and how
much does a WARN remove?

Sets (downloaded once into data/external/, from the dataset revision pinned
below; a different revision is refused):
  notinject  NotInject (Li & Liu, arXiv 2410.22770, MIT): 339 innocent
             prompts built around injection trigger words ("ignore", "system").
  enron      2,000 emails (40 random places of 50 consecutive, seed 0) from the Enron corpus
             (corbt/enron-emails, raw bodies): real requests between
             colleagues, i.e. orders that are not attacks.
  tatoeba    1,000 ordinary sentences each in Russian, Bokmål, Nynorsk.
  supertanks the innocent messages of Super Tanks' red-team corpus.

Each text is checked two ways, as Super Tanks checks it:
  tool output   scan_message + scan_tool_result (detectors on), as the gateway
                does; Enron emails as an email object (subject, from, to,
                date, body), the rest as plain text.
  user message  scan_message only (detectors never run on user messages), with
                the current regex filter and with the one on Super Tanks'
                main branch (before the normalizer), to see what the
                normalizer changes.

Usage (from this folder, PYTHONPATH = repository root):
  python eval_false_alarms.py                 # report + data/false_alarms.json
"""

import argparse
import importlib.util
import json
import math
import os
import random
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

os.environ.setdefault("ST_ZEF_DETECTORS", "1")

from core.security import zef_combine  # noqa: E402
from core.security.zef_injection_filter import FilterVerdict, scan_message  # noqa: E402

import eval_phrase_similarity as eps  # noqa: E402

HERE = Path(__file__).resolve().parent
EXTERNAL = HERE / "data" / "external"
ROWS_API = "https://datasets-server.huggingface.co/rows"
INFO_API = "https://huggingface.co/api/datasets/"
NOTINJECT = ("leolee99/NotInject", "847ae76cf8fea5ed325429e569ae8cfef022d2e0",
             ("NotInject_one", "NotInject_two", "NotInject_three"))
ENRON = ("corbt/enron-emails", "cfc06c758093d90993abce1a43668fb7357258a6", 517401)
ENRON_SAMPLE, ENRON_PAGE = 2000, 50  # 40 random pages of 50 consecutive emails
MAX_CHARS = 30000  # as the dataset's longest tool outputs


def _get(url, attempts=6):
    """JSON from the URL; waits between calls and backs off on errors (the rows
    API rate-limits)."""
    for i in range(attempts):
        time.sleep(1.5)
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.loads(r.read())
        except OSError:
            if i == attempts - 1:
                raise
            time.sleep(10 * 2 ** i)


def _check_revision(repo, revision):
    current = _get(INFO_API + repo)["sha"]
    if current != revision:
        raise SystemExit(f"{repo} is at revision {current}, pinned {revision}: "
                         "the served rows may differ; check and re-pin")


def _rows(repo, split, offset, length):
    q = urllib.parse.urlencode({"dataset": repo, "config": "default", "split": split,
                                "offset": offset, "length": length})
    return [r["row"] for r in _get(f"{ROWS_API}?{q}")["rows"]]


def _cached(name, fetch):
    path = EXTERNAL / name / "rows.jsonl"
    if not path.exists():
        rows = fetch()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                        encoding="utf-8")
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def notinject():
    repo, revision, splits = NOTINJECT

    def fetch():
        _check_revision(repo, revision)
        return [r for s in splits for off in (0, 100) for r in _rows(repo, s, off, 100)]
    return [r["prompt"] for r in _cached("notinject", fetch)]


def enron():
    repo, revision, total = ENRON

    def fetch():
        _check_revision(repo, revision)
        starts = random.Random(0).sample(range(0, total - ENRON_PAGE, ENRON_PAGE),
                                         ENRON_SAMPLE // ENRON_PAGE)
        return [r for s in sorted(starts) for r in _rows(repo, "train", s, ENRON_PAGE)]
    out = []
    for r in _cached("enron", fetch):
        if r["body"].strip() and len(r["body"]) <= MAX_CHARS:
            out.append({"subject": r["subject"], "from": r["from"], "to": r["to"],
                        "date": r["date"], "body": r["body"]})
    return out


def supertanks_clean():
    from tests.security.redteam.corpus import CLEAN_CASES
    return [text for text, _, _ in CLEAN_CASES]


def main_branch_filter():
    """scan_message of Super Tanks' main branch (before the normalizer)."""
    root = Path(zef_combine.__file__).resolve().parents[2]
    src = subprocess.run(["git", "show", "main:core/security/zef_injection_filter.py"],
                         cwd=root, capture_output=True, text=True, encoding="utf-8",
                         check=True).stdout
    path = Path(tempfile.mkdtemp()) / "zef_filter_main.py"
    path.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("zef_filter_main", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.scan_message


def wilson_upper(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    return 100 * ((p + z * z / (2 * n)) + z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d


def text_of(value):
    """A tool result flattened to text, as core/gateway.py's _extract_text does."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(text_of(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return " ".join(text_of(v) for v in value)
    return "" if value is None or isinstance(value, (int, float, bool)) else str(value)


def as_tool_output(item):
    flat = text_of(item)
    v = scan_message(flat, source="tool_output:eval")
    v = zef_combine.scan_tool_result(item, v)
    removed = 0.0
    if v.verdict is FilterVerdict.WARN and v.remove:
        kept = zef_combine.remove_sentences(item, v.remove)
        if kept is None:
            v.verdict = FilterVerdict.BLOCK  # as the gateway: unremovable -> withheld
        else:
            words = len(flat.split())
            removed = 1 - len(text_of(kept).replace(zef_combine.REMOVED, " ").split()) / max(words, 1)
    return v, removed


def report(name, items, old_scan):
    n = len(items)
    tool = [as_tool_output(it) for it in items]
    warn = sum(v.verdict is FilterVerdict.WARN for v, _ in tool)
    block = sum(v.verdict is FilterVerdict.BLOCK for v, _ in tool)
    flagged = warn + block
    removed = sorted(r for v, r in tool if v.verdict is FilterVerdict.WARN)
    by = Counter(p for v, _ in tool if v.verdict is not FilterVerdict.PASS for p in v.matched_patterns)
    user_new = sum(scan_message(text_of(it), source="telegram:user").verdict is not FilterVerdict.PASS
                   for it in items)
    user_old = sum(old_scan(text_of(it), source="telegram:user").verdict.value != "pass"
                   for it in items)
    print(f"\n{name}: {n} texts")
    print(f"  as tool output: flagged {flagged} of {n} ({100 * flagged / n:.1f} %, "
          f"95 % upper bound {wilson_upper(flagged, n):.1f} %): WARN {warn}, BLOCK {block}")
    if removed:
        print(f"    text removed by a WARN: median {100 * removed[len(removed) // 2]:.0f} %, "
              f"most {100 * removed[-1]:.0f} %")
    if by:
        print(f"    signals: {dict(by.most_common())}")
    print(f"  as user message: flagged {user_new} of {n} (main branch, before the normalizer: {user_old})")
    examples = [{"text": text_of(it)[:600], "verdict": v.verdict.value, "patterns": v.matched_patterns,
                 "segments": v.flagged_segments, "removed_share": round(r, 2)}
                for it, (v, r) in zip(items, tool) if v.verdict is not FilterVerdict.PASS]
    return {"n": n, "warn": warn, "block": block, "user_new": user_new, "user_old": user_old,
            "flagged": examples}


def main(argv=None):
    argparse.ArgumentParser(description=__doc__.split("\n\n")[0]).parse_args(argv)
    if zef_combine._load() is None:
        raise SystemExit("detectors unavailable (see the log); install super-tanks[zef] and the model")
    old_scan = main_branch_filter()
    sets = {"notinject": notinject(), "enron": enron(), "supertanks": supertanks_clean()}
    sets.update({f"tatoeba_{lang}": texts for lang, texts in eps.tatoeba_sentences().items()})
    out = {name: report(name, items, old_scan) for name, items in sets.items()}
    path = HERE / "data" / "false_alarms.json"
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nflagged texts: {path}")


if __name__ == "__main__":
    main()
