"""
eval_outputs.py
===============
Single-output test (../README.md): does a system flag attacked tool
outputs and leave clean ones alone, on the main split's test part?

A system is a function from rows to a boolean array (flagged = WARN or BLOCK).
Detector candidates add a scoring function and get their threshold from
threshold_at_budget() on clean training outputs; the regex filter is built in.

Reports, with 95 % bootstrap intervals: attacks caught and false alarms overall,
attacks caught per attack style (framing), false alarms per kind of clean row.
Results are also written to data/eval_outputs_<system>.json.

Usage:
  python eval_outputs.py --system regex
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "zef_pairs.jsonl"

# Main split: AgentDojo by suite; InjecAgent attacks by goal group (a fixed,
# hash-chosen share of the attacker goals per intent is never trained on),
# InjecAgent clean rows by text; BIPIA by its own train/test split.
TRAIN_SUITES, TEST_SUITES = {"workspace", "travel"}, {"banking", "slack"}
EXTERNAL_TEST_SHARE = 0.3
BOOTSTRAP = 1000


def load_rows(path=DATA):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def row_id(r):
    """Stable id of a row (content hash)."""
    return hashlib.sha256((r["user_request"] + "\0" + r["tool_output"])
                          .encode("utf-8")).hexdigest()[:16]


def _stable_unit(text):
    """Deterministic number in [0, 1) for a string."""
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16) / 16 ** 8


def _external_part(rows):
    """'train' / 'test' for external rows, None for AgentDojo rows."""
    groups = {}
    for r in rows:
        if r["benchmark"] == "injecagent" and r["label"]:
            groups.setdefault(r["intent"], set()).add(r["goal_group"])
    test_groups = set()
    for gs in groups.values():
        ordered = sorted(gs, key=_stable_unit)
        test_groups.update(ordered[:round(EXTERNAL_TEST_SHARE * len(ordered))])
    out = []
    for r in rows:
        if r["benchmark"] == "injecagent":
            test = (r["goal_group"] in test_groups if r["label"]
                    else _stable_unit(r["tool_output"]) < EXTERNAL_TEST_SHARE)
            out.append("test" if test else "train")
        elif r["benchmark"] == "bipia":
            out.append(r["user_task_id"].split(":", 1)[0])
        else:
            out.append(None)
    return out


def main_split(rows):
    """(train, test) boolean masks; aborts if an attacker goal is in both."""
    part = _external_part(rows)
    train = np.array([(r["benchmark"] == "agentdojo" and r["suite"] in TRAIN_SUITES)
                      or p == "train" for r, p in zip(rows, part)])
    test = np.array([(r["benchmark"] == "agentdojo" and r["suite"] in TEST_SUITES)
                     or p == "test" for r, p in zip(rows, part)])
    shared = ({r["goal_group"] for r, t in zip(rows, train) if t and r["label"]}
              & {r["goal_group"] for r, t in zip(rows, test) if t and r["label"]}) - {None}
    if shared or (train & test).any():
        raise SystemExit(f"main split: goal groups in both parts: {sorted(shared)[:5]}")
    return train, test


def clean_kind(r):
    """What a clean row is: twin, control, look-alike, env_read, trace, ..."""
    f = r["framing"] or ""
    if f.startswith("control_injecagent_"):
        kind = "control_" + f[len("control_injecagent_"):]
    elif f.startswith("bipia_harmless"):
        kind = "harmless_hijack"
    elif f.startswith("control_override"):
        kind = "lookalike"
    elif f.startswith("control_"):
        kind = "control"
    else:
        kind = {"groundtruth": "twin"}.get(r["source"], r["source"])
    return f"{r['benchmark']}:{kind}"


def threshold_at_budget(scores, clean_train, budget):
    """Threshold above which `budget` of the clean training outputs score."""
    return float(np.quantile(scores[clean_train], 1 - budget))


def rate_ci(flags, rng):
    """Rate with a 95 % bootstrap interval."""
    n = len(flags)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    boots = [flags[rng.integers(0, n, n)].mean() for _ in range(BOOTSTRAP)]
    return float(flags.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def regex_system(rows):
    from core.security.zef_injection_filter import FilterVerdict, scan_message
    return np.array([scan_message(r["tool_output"], source="tool_output:eval").verdict
                     is not FilterVerdict.PASS for r in rows])


SYSTEMS = {"regex": regex_system}


def report(name, flagged, rows, test):
    y = np.array([r["label"] for r in rows])
    attacked, clean = test & (y == 1), test & (y == 0)
    rng = np.random.default_rng(0)
    out = {"system": name, "attacked": int(attacked.sum()), "clean": int(clean.sum())}

    def line(label, mask):
        k, n = int((flagged & mask).sum()), int(mask.sum())
        rate, lo, hi = rate_ci(flagged[mask], rng)
        print(f"  {label:<44} {k:>6} of {n:<6} {rate:6.1%}  [{lo:.1%}, {hi:.1%}]")
        return {"flagged": k, "of": n, "rate": rate, "ci": [lo, hi]}

    print(f"\n{name}: main split, {out['attacked']} attacked and {out['clean']} clean tool outputs")
    out["attacks_caught"] = line("attacks caught", attacked)
    out["false_alarms"] = line("false alarms", clean)
    print("\n  attacks caught, per attack style")
    styles = sorted({rows[i]["framing"] or "-" for i in np.flatnonzero(attacked)})
    out["by_style"] = {s: line(s, attacked & np.array([(r["framing"] or "-") == s for r in rows]))
                       for s in styles}
    print("\n  false alarms, per kind of clean row")
    kinds = np.array([clean_kind(r) for r in rows])
    out["by_kind"] = {k: line(k, clean & (kinds == k)) for k in sorted(set(kinds[clean]))}
    (HERE / "data" / f"eval_outputs_{name}.json").write_text(json.dumps(out, indent=1),
                                                            encoding="utf-8")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Single-output test on the main split.")
    p.add_argument("--system", choices=SYSTEMS, default="regex")
    args = p.parse_args(argv)
    rows = load_rows()
    _, test = main_split(rows)
    report(args.system, SYSTEMS[args.system](rows), rows, test)


if __name__ == "__main__":
    main()
