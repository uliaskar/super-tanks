"""
eval_combine.py
===============
Step 7 (../README.md): the verdict combiner. One verdict per tool output from
the regex filter and three detectors: phrase similarity (B, MiniLM), phrase
matcher (A2, word alignment) and instruction detector (C2c, character
fragments).

Any-flag: each detector has two thresholds. Above the high one it flags; above
the low one it gives a medium signal, and two medium signals together also
flag (the regex filter's flag counts as one). A fifth signal, H ("an order
inside hidden data"), is C2c's highest score among the segments that came
from hidden data (zef_segment.segment_origins); it is a medium signal only.
The thresholds are searched jointly on the training part: at most BUDGET of
the clean training outputs flagged in total, regex filter included, with the
most training attacks caught. B's and C2c's own flags are fixed (HIGH_SHARES);
the training data lacks attacks some flags exist for, such as Russian framings. C2c's training scores are held out per training group (each group
scored by a model trained without it); B and A2 are not trained.

Verdict: BLOCK if the regex filter blocks or at least two flags agree (the
regex filter's flag and each detector's high flag count as one each); WARN on
one flag or two medium signals; else PASS. Never looser than the regex verdict.

Modes:
  --split main    main split: caught / false alarms, BLOCK tier, Tatoeba
  --robustness    robustness test of the whole system
  --latency       time per tool output

Usage:
  python eval_combine.py --split main --robustness --latency
"""

import argparse
import itertools
import random
import time

import numpy as np
from core.security import zef_segment
from core.security.zef_injection_filter import FilterVerdict, scan_message

import eval_instruction as ei
import eval_outputs as eo
import eval_phrase_match as pm
import eval_phrase_similarity as eps
import phrase_list

BUDGET = 0.01
B = "B2-paraphrase-multilingual-minilm"
# Candidate thresholds, as the share of clean training outputs scoring above them.
# B's and C2c's own flags stay at 0.25 %: searched freely, B's moved up and let
# 30 of 88 Russian-framed robustness seeds escape (the training part has few
# such attacks to lose), and C2c's moved up and lost varied plain orders.
HIGH_SHARES = (0.0025,)
LOW_SHARES = (None, 0.01, 0.02, 0.05, 0.1)
# H's low thresholds: shares of the training outputs with hidden data scoring below
HIDDEN_QUANTILES = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9)
NO_SEGMENT = -1e8  # SegmentCandidate.score gives -1e9 to an empty segment list
PASS, WARN, BLOCK = 0, 1, 2
_LEVEL = {FilterVerdict.PASS: PASS, FilterVerdict.WARN: WARN, FilterVerdict.BLOCK: BLOCK}


def regex_verdicts(texts):
    return np.array([_LEVEL[scan_message(t, source="tool_output:eval").verdict] for t in texts])


def hidden_segments(texts):
    """Per text: its segments that came from hidden data."""
    return [[s for s, hidden in zef_segment.segment_origins(t) if hidden] for t in texts]


def combine(regex, scores, thresholds):
    """thresholds: detector -> (high, low)."""
    flags = (regex > PASS).astype(int)
    medium = flags.copy()
    for k, s in scores.items():
        high, low = thresholds[k]
        flags += s > high
        medium += s > low
    return np.where((regex == BLOCK) | (flags >= 2), BLOCK,
                    np.where((flags >= 1) | (medium >= 2), WARN, PASS))


def search(regex, scores, y, train):
    """Thresholds within the budget that catch the most training attacks."""
    clean, attacked = train & (y == 0), train & (y == 1)
    cap = int(BUDGET * clean.sum())

    def by_share(s, shares):
        return [np.inf if q is None else float(np.quantile(s[clean], 1 - q)) for q in shares]

    a2_values = sorted({float(v) for v in scores["A2"][train] if v >= 0.5})
    with_hidden = scores["H"][train & (scores["H"] > NO_SEGMENT)]
    options = {k: (by_share(scores[k], HIGH_SHARES), by_share(scores[k], LOW_SHARES))
               for k in ("B", "C2c")}
    options["A2"] = (a2_values, [np.inf] + a2_values)
    options["H"] = ([np.inf], [np.inf] + [float(np.quantile(with_hidden, q)) for q in HIDDEN_QUANTILES])
    pairs = {k: [(h, lo) for h in highs for lo in lows if lo <= h or lo == np.inf]
             for k, (highs, lows) in options.items()}
    best, best_key = None, None
    for combo in itertools.product(*pairs.values()):
        t = dict(zip(pairs, combo))
        flagged = combine(regex, scores, t) > PASS
        key = (flagged[attacked].sum(), -flagged[clean].sum())
        if flagged[clean].sum() <= cap and (best_key is None or key > best_key):
            best, best_key = t, key
    if best is None:
        raise SystemExit(f"no thresholds keep the clean training outputs flagged within {cap}")
    return best


class Combiner:
    """The fitted system: a list of tool outputs -> verdicts. Also a
    robustness-test system (caught if any output is not PASS)."""

    def __init__(self, c2c, thresholds):
        self.c2c, self.thresholds, self.a2 = c2c, thresholds, pm.matcher("A2")

    def scores(self, texts):
        phrase_segs = eps.segments_of(texts)
        return {"B": eps.scores(B, phrase_segs), "A2": pm.scores(self.a2, phrase_segs),
                "C2c": self.c2c.score(ei.segments_of(texts)),
                "H": self.c2c.score(hidden_segments(texts))}

    def verdicts(self, texts):
        return combine(regex_verdicts(texts), self.scores(texts), self.thresholds)

    def prepare(self, outputs):  # embed every segment of the test in one batch
        eps.scores(B, [[s for ss in eps.segments_of(outputs) for s in ss]])

    def __call__(self, outputs):
        return bool((self.verdicts(outputs) > PASS).any())


def fit():
    """(data, test, regex verdicts, test-time scores, fitted Combiner)."""
    data = ei.Data()
    train, test = eo.main_split(data.rows)
    regex = regex_verdicts(data.texts)
    phrase_segs = eps.segments_of(data.texts)
    hidden = hidden_segments(data.texts)
    c2c = ei.fit_candidate("C2c", data, train)
    scores = {"B": eps.scores(B, phrase_segs), "A2": pm.scores(pm.matcher("A2"), phrase_segs),
              "C2c": ei.score_rows(c2c, data), "H": c2c.score(hidden)}
    held = {"C2c": scores["C2c"].copy(), "H": scores["H"].copy()}
    for g in sorted(set(data.group[train])):
        inner = train & (data.group == g)
        idx = np.flatnonzero(inner)
        model = ei.fit_candidate("C2c", data, train & ~inner)
        held["C2c"][idx] = ei.score_rows(model, data, inner)[idx]
        held["H"][idx] = model.score([hidden[i] for i in idx])
    thresholds = search(regex, {**scores, **held}, data.y, train)
    return data, test, regex, scores, Combiner(c2c, thresholds)


def imitation(r):
    """Clean text written to look like an attack (controls, look-alikes, harmless hijacks)."""
    return not r["label"] and eo.clean_kind(r).split(":")[-1].startswith(
        ("control", "lookalike", "harmless"))


def main_split(data, test, regex, scores, comb):
    print("thresholds (high, low):",
          {k: tuple(round(x, 3) for x in v) for k, v in comb.thresholds.items()})
    v = combine(regex, scores, comb.thresholds)
    eo.report("combined", v > PASS, data.rows, test)
    y = data.y
    imit = np.array([imitation(r) for r in data.rows])
    for name, f in (("any flag (WARN or BLOCK)", v > PASS), ("BLOCK", v == BLOCK)):
        print(f"  {name:<26} caught {f[test & (y == 1)].sum()} of {(test & (y == 1)).sum()}; "
              f"false alarms on imitation text {f[test & imit].sum()} of {(test & imit).sum()}, "
              f"on real environment output {f[test & (y == 0) & ~imit].sum()} of {(test & (y == 0) & ~imit).sum()}")
    for lang, texts in eps.tatoeba_sentences().items():
        print(f"  Tatoeba {lang}: {int((comb.verdicts(texts) > PASS).sum())} of {len(texts)} flagged")


def robustness(comb):
    import eval_robustness as er
    er.check_held_out(phrase_list.expanded())
    er.run(comb, "combined")


def latency(comb, rows, n=200):
    """Regex filter, segmenting, A2, C2c and H in this process; B's MiniLM
    embedding is timed separately (it runs in the embedding environment)."""
    sample = random.Random(0).sample([r["tool_output"] for r in rows], n)
    pm.edits.cache_clear()
    times = []
    for text in sample:
        t = time.perf_counter()
        regex_verdicts([text])
        segs = eps.segments_of([text])
        pm.scores(comb.a2, segs)
        comb.c2c.score(ei.segments_of([text]))
        comb.c2c.score(hidden_segments([text]))
        times.append((time.perf_counter() - t) * 1000)
    times.sort()
    print(f"regex + segmenting + A2 + C2c + H: ms per tool output: median {times[n // 2]:.1f}, "
          f"95th percentile {times[int(0.95 * (n - 1))]:.1f}, max {times[-1]:.0f}")
    eps.latency([B], rows, n)


def main(argv=None):
    p = argparse.ArgumentParser(description="Verdict combiner (step 7).")
    p.add_argument("--split", choices=["main"])
    p.add_argument("--robustness", action="store_true")
    p.add_argument("--latency", action="store_true")
    args = p.parse_args(argv)
    data, test, regex, scores, comb = fit()
    if args.split == "main":
        main_split(data, test, regex, scores, comb)
    if args.robustness:
        robustness(comb)
    if args.latency:
        latency(comb, data.rows)


if __name__ == "__main__":
    main()
