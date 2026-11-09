"""
check_runtime.py
================
Step 8 (../README.md): does the Super Tanks runtime (core/security/
zef_combine.py, with its shipped parameter files and the downloaded model)
give the same verdict as the harness (eval_combine.py) on every tool output
of the main split's test part? Also reports the largest score difference per
detector and the runtime's time per tool output.

Usage:
  python check_runtime.py [--latency 200]
"""

import argparse
import random
import time

import numpy as np
from core.security import zef_combine as zc
from core.security.zef_injection_filter import FilterVerdict, scan_message

import eval_combine as ec

_LEVEL = {FilterVerdict.PASS: ec.PASS, FilterVerdict.WARN: ec.WARN, FilterVerdict.BLOCK: ec.BLOCK}


def main(argv=None):
    p = argparse.ArgumentParser(description="Runtime vs harness verdicts (step 8).")
    p.add_argument("--latency", type=int, default=200, help="tool outputs to time")
    args = p.parse_args(argv)
    data, test, regex, scores, comb = ec.fit()
    harness = ec.combine(regex, scores, comb.thresholds)
    runtime = zc.Detectors()
    idx = np.flatnonzero(test)
    differ, worst = [], {k: 0.0 for k in zc.NAMES}
    for i in idx:
        text = data.texts[i]
        r = scan_message(text, source="tool_output:eval")
        s = runtime.scores(text)
        verdict, _, _ = zc.combine(r.verdict, s, runtime.thresholds)
        for k, (score, _) in s.items():
            worst[k] = max(worst[k], abs(score - scores[k][i]))
        if _LEVEL[verdict] != harness[i]:
            differ.append(i)
    print(f"test-part tool outputs: {len(idx)}; verdict differs from the harness: {len(differ)}")
    print("largest score difference per detector:", {zc.NAMES[k]: f"{v:.1e}" for k, v in worst.items()})
    for i in differ[:5]:
        print(f"  differs: row {i}, harness {harness[i]}, framing {data.rows[i]['framing']}")
    sample = random.Random(0).sample([data.texts[i] for i in idx], args.latency)
    times = []
    for text in sample:
        t = time.perf_counter()
        runtime.scores(text)
        scan_message(text, source="tool_output:eval")
        times.append((time.perf_counter() - t) * 1000)
    times.sort()
    n = len(times)
    print(f"runtime, ms per tool output ({runtime.similarity._session.get_providers()[0]}): "
          f"median {times[n // 2]:.0f}, 95th percentile {times[int(0.95 * (n - 1))]:.0f}, max {times[-1]:.0f}")


if __name__ == "__main__":
    main()
