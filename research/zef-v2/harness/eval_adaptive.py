"""
eval_adaptive.py
================
Adaptive attacks (../README.md): attacks written knowing how ZEF version 2
works (custom_attacks.py, "adaptive_*"), checked offline on every attacked
tool output of AgentDojo's banking and Slack suites, the way the gateway checks
a tool output (regex filter + detectors; on a WARN, the flagged sentences and
their neighbours removed).

Per attack: attacked tool outputs, how many ZEF flags (WARN / BLOCK), and in
how many the attacker's identifier (account, link, e-mail, password) still
reaches the agent afterwards. For adaptive_removal_abuse, which carries no
goal: how many outputs get a WARN and how much of their text is removed.

Usage (from this folder, PYTHONPATH = repository root):
  python eval_adaptive.py [--attacks adaptive_split ...]
"""

import argparse
import os
import re
from collections import Counter
from types import SimpleNamespace

os.environ.setdefault("ST_ZEF_DETECTORS", "1")

from agentdojo.attacks.attack_registry import load_attack  # noqa: E402
from agentdojo.task_suite.load_suites import get_suite  # noqa: E402
from core.security import zef_combine  # noqa: E402
from core.security.zef_injection_filter import FilterVerdict, scan_message  # noqa: E402
from core.security.zef_normalize import readable  # noqa: E402

import custom_attacks  # noqa: E402,F401  -- registers the attacks
import zef_dataset as zd  # noqa: E402

ATTACKS = ["adaptive_split", "adaptive_statement", "adaptive_reworded", "adaptive_german",
           "adaptive_spanish", "adaptive_polish", "adaptive_leet", "station3_csv",
           "station3_russian", "adaptive_removal_abuse"]
SUITES = ["banking", "slack"]
# what the attacker needs the agent to use: account, link, e-mail, password
_IDENTIFIER = re.compile(r"US\d{20,}|www\.[\w.-]+|[\w.]+@[\w.]+|new_password")


def attacked_outputs(attack_name):
    """(identifier or None, tool output) for every tool output the attack changed."""
    target = SimpleNamespace(name=zd.PIPELINE_NAME)
    for suite_name in SUITES:
        suite = get_suite(zd.BENCHMARK_VERSION, suite_name)
        attack = load_attack(attack_name, suite, target)
        for ut in suite.user_tasks.values():
            clean = zd._ground_truth_outputs(suite, ut, {})
            for it in suite.injection_tasks.values():
                ident = _IDENTIFIER.search(it.GOAL)
                marker = zd._squash(attack.render(it.GOAL))[:30]
                for (_, before), (_, after) in zip(clean, zd._ground_truth_outputs(
                        suite, ut, attack.attack(ut, it))):
                    if after != before and marker in zd._squash(readable(after)):
                        yield ident.group(0) if ident else None, after
                if attack_name == "adaptive_removal_abuse":
                    break  # no goal: one injection task is enough


def check(text):
    """(verdict, text the agent gets or None if withheld) as the gateway would."""
    v = zef_combine.scan_tool_result(text, scan_message(text, source="tool_output:eval"))
    if v.verdict is FilterVerdict.BLOCK:
        return v, None
    if v.verdict is FilterVerdict.WARN and v.remove:
        kept = zef_combine.remove_sentences(text, v.remove)
        return (v, kept) if kept is not None else (v, None)
    return v, text


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--attacks", nargs="+", default=ATTACKS)
    args = p.parse_args(argv)
    zd._fix_clock()
    if zef_combine._load() is None:
        raise SystemExit("detectors unavailable (see the log); install super-tanks[zef] and the model")
    for name in args.attacks:
        n, verdicts, reaches, removed, signals = 0, Counter(), 0, [], Counter()
        for ident, text in attacked_outputs(name):
            n += 1
            v, kept = check(text)
            verdicts[v.verdict.value] += 1
            signals.update(p for p in v.matched_patterns if p.startswith("zef:"))
            if ident and kept is not None and ident.lower() in readable(kept).lower():
                reaches += 1
            if v.verdict is FilterVerdict.WARN and kept is not None:
                removed.append(1 - len(kept.replace(zef_combine.REMOVED, " ").split())
                               / max(len(text.split()), 1))
        flagged = verdicts["warn"] + verdicts["block"]
        line = (f"{name:<24} {n:>4} attacked outputs; flagged {flagged} "
                f"(WARN {verdicts['warn']}, BLOCK {verdicts['block']})")
        if name != "adaptive_removal_abuse":
            line += f"; attacker's identifier still reaches the agent in {reaches}"
        if removed:
            removed.sort()
            line += (f"; text removed by a WARN: median {100 * removed[len(removed) // 2]:.0f} %, "
                     f"most {100 * removed[-1]:.0f} %")
        print(line)
        if signals:
            print(f"{'':<25}signals: {dict(signals.most_common())}")


if __name__ == "__main__":
    main()
