"""
eval_phrase_match.py
====================
Step 5 (../README.md): candidates for the phrase matcher, i.e. recognising
known injection phrases despite spelling errors. No model and no training: a
candidate scores how much of a known phrase (phrase_list.expanded(), the same
list phrase similarity uses, phrases of MIN_WORDS or more words) appears in a
segment, despite misspellings. Shorter phrases ("System prompt:") match whole
inside ordinary sentences and are left to phrase similarity (step 7).

Candidates:
  A1  letter-trigram share: share of the phrase's trigrams found in the segment
  A2  word alignment: phrase words matched in order to segment words within an
      edit-distance tolerance (transposition = 1 edit); share of phrase words
      matched inside a window of the phrase's length + 2 words
A tool output's score is the highest over its segments and phrases; it is
flagged above a threshold set so that BUDGET of the clean training outputs
score above it. Segments as for phrase similarity (sentences and clauses).

Modes:
  --split main    main split, alone and with the regex filter, plus Tatoeba
  --robustness    robustness test, threshold from the main split
  --latency       scoring time per tool output

Usage:
  python eval_phrase_match.py --split main --robustness --latency
"""

import argparse
import random
import re
import time
from functools import lru_cache

import numpy as np

import eval_outputs as eo
import eval_phrase_similarity as eps
import phrase_list

BUDGET = 0.01
MIN_WORDS = 3
CANDIDATES = ["A1", "A2"]
_WORD = re.compile(r"\w+")


def words(text):
    return _WORD.findall(text.lower())


# ── A1: letter-trigram share ──────────────────────────────────────────────
def trigrams(text):
    out = set()
    for w in words(text):
        w = f" {w} "
        out.update(w[i:i + 3] for i in range(len(w) - 2))
    return out


class TrigramMatcher:
    def __init__(self, phrases):
        self.phrases = [trigrams(p) for p in phrases]

    def score(self, segment):
        seg = trigrams(segment)
        return max(len(p & seg) / len(p) for p in self.phrases)


# ── A2: word alignment with edit-distance tolerance ───────────────────────
def tolerance(word):
    return 0 if len(word) <= 3 else 1 if len(word) <= 6 else 2


@lru_cache(maxsize=1_000_000)
def edits(a, b):
    """Optimal-string-alignment distance: insert, delete, substitute, swap neighbours."""
    if abs(len(a) - len(b)) > 2:
        return 3
    prev2, prev = None, list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cost = a[i - 1] != b[j - 1]
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                cur[j] = min(cur[j], prev2[j - 2] + 1)
        prev2, prev = prev, cur
    return prev[-1]


def deletions(word, depth):
    """The word and every string made by deleting up to `depth` letters."""
    out, layer = {word}, {word}
    for _ in range(depth):
        layer = {w[:i] + w[i + 1:] for w in layer for i in range(len(w))}
        out |= layer
    return out


class AlignmentMatcher:
    def __init__(self, phrases):
        self.phrases = [words(p) for p in phrases]
        self.vocabulary = sorted({w for p in self.phrases for w in p})
        # Deletion neighbourhoods: each edit (substitution, insertion, deletion,
        # neighbour swap) is undone by at most one deletion on each side, so a
        # segment word within a phrase word's tolerance shares a deletion
        # string with it; only those candidates get the exact distance.
        self._by_deletion = {}
        for pw in self.vocabulary:
            for d in deletions(pw, tolerance(pw)):
                self._by_deletion.setdefault(d, set()).add(pw)
        self._longest = max(map(len, self.vocabulary))
        self._cache = {}

    def _matches(self, word):
        """Phrase words this segment word counts as (cached per word)."""
        if word not in self._cache:
            if len(word) > self._longest + 2:  # more than 2 edits from every phrase word
                self._cache[word] = frozenset()
                return self._cache[word]
            candidates = set().union(*(self._by_deletion.get(d, ()) for d in deletions(word, 2)))
            self._cache[word] = frozenset(
                pw for pw in candidates if edits(pw, word) <= tolerance(pw))
        return self._cache[word]

    @staticmethod
    def _aligned(phrase, window):
        """Most phrase words matched in order (longest common subsequence)."""
        prev = [0] * (len(window) + 1)
        for pw in phrase:
            cur = [0]
            for j, matches in enumerate(window):
                cur.append(prev[j] + 1 if pw in matches else max(prev[j + 1], cur[j]))
            prev = cur
        return prev[-1]

    def score(self, segment):
        seg = [self._matches(w) for w in words(segment)]
        present = frozenset().union(*seg)
        best = 0.0
        for phrase in self.phrases:
            if sum(pw in present for pw in phrase) / len(phrase) <= best:
                continue  # cannot beat the best score so far
            size = len(phrase) + 2
            # an optimal window can always start at a matched word
            for start in (j for j, m in enumerate(seg) if m & set(phrase)):
                best = max(best, self._aligned(phrase, seg[start:start + size]) / len(phrase))
            if best == 1.0:
                return best
        return best


def matcher(name):
    phrases = [p for p in phrase_list.expanded() if len(p.split()) >= MIN_WORDS]
    return TrigramMatcher(phrases) if name == "A1" else AlignmentMatcher(phrases)


def scores(m, segment_lists):
    unique = {s: m.score(s) for s in {s for ss in segment_lists for s in ss}}
    return np.array([max((unique[s] for s in ss), default=0.0) for ss in segment_lists])


def main_split(candidates, rows):
    y = np.array([r["label"] for r in rows])
    train, test = eo.main_split(rows)
    seg_rows = eps.segments_of([r["tool_output"] for r in rows])
    regex = eo.regex_system(rows)
    tatoeba = eps.tatoeba_sentences()
    for name in candidates:
        m = matcher(name)
        s = scores(m, seg_rows)
        thr = eo.threshold_at_budget(s, train & (y == 0), BUDGET)
        print(f"{name}: threshold {thr:.3f}")
        eo.report(f"match-{name}", s > thr, rows, test)
        eo.report(f"regex+match-{name}", (s > thr) | regex, rows, test)
        for lang, texts in tatoeba.items():
            t = scores(m, eps.segments_of(texts))
            print(f"  Tatoeba {lang}: {int((t > thr).sum())} of {len(texts)} flagged")


class MatchSystem:
    """A candidate as a robustness-test system: caught if any tool output
    scores above the threshold."""

    def __init__(self, m, threshold):
        self.m, self.threshold = m, threshold

    def __call__(self, outputs):
        return bool((scores(self.m, eps.segments_of(outputs)) > self.threshold).any())


def robustness(candidates, rows):
    import eval_robustness as er
    er.check_held_out(phrase_list.expanded())
    y = np.array([r["label"] for r in rows])
    train, _ = eo.main_split(rows)
    seg_rows = eps.segments_of([r["tool_output"] for r in rows])
    for name in candidates:
        m = matcher(name)
        thr = eo.threshold_at_budget(scores(m, seg_rows), train & (y == 0), BUDGET)
        er.run(MatchSystem(m, thr), f"match-{name}")


def latency(candidates, rows, n=200):
    sample = random.Random(0).sample([r["tool_output"] for r in rows], n)
    for name in candidates:
        m = matcher(name)
        edits.cache_clear()
        times = []
        for text in sample:
            t = time.perf_counter()
            scores(m, eps.segments_of([text]))
            times.append((time.perf_counter() - t) * 1000)
        times.sort()
        print(f"{name}: ms per tool output, segmenting included: median {times[n // 2]:.1f}, "
              f"95th percentile {times[int(0.95 * (n - 1))]:.1f}, max {times[-1]:.0f}")


def main(argv=None):
    p = argparse.ArgumentParser(description="Phrase-matcher candidates (step 5).")
    p.add_argument("--split", choices=["main"])
    p.add_argument("--robustness", action="store_true")
    p.add_argument("--latency", action="store_true")
    p.add_argument("--candidates", nargs="+", choices=CANDIDATES, default=CANDIDATES)
    args = p.parse_args(argv)
    rows = eo.load_rows()
    if args.split == "main":
        main_split(args.candidates, rows)
    if args.robustness:
        robustness(args.candidates, rows)
    if args.latency:
        latency(args.candidates, rows)


if __name__ == "__main__":
    main()
