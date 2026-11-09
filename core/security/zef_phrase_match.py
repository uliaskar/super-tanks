"""
core/security/zef_phrase_match.py
==================================
ZEF phrase matcher: finds known injection phrases despite spelling errors.

A segment's score is the largest share of a known phrase's words found in it
in order, inside a window of the phrase's length + 2 words. A segment word
counts as a phrase word within an edit-distance tolerance: 0 edits for words
of up to 3 letters, 1 for 4–6, 2 for longer; an edit is a substitution,
insertion, deletion or swap of neighbouring letters. Phrases of fewer than
`min_words` words are left out: they match whole inside ordinary sentences
("The system prompt on the kiosk screens was updated").

Pure Python, no model. Measured in research/zef-v2 (eval_phrase_match.py).
"""

import re
from functools import lru_cache

_WORD = re.compile(r"\w+")


def words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def tolerance(word: str) -> int:
    return 0 if len(word) <= 3 else 1 if len(word) <= 6 else 2


@lru_cache(maxsize=200_000)
def edits(a: str, b: str) -> int:
    """Optimal-string-alignment distance (3 means "more than 2")."""
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


def deletions(word: str, depth: int) -> set[str]:
    """The word and every string made by deleting up to `depth` letters."""
    out, layer = {word}, {word}
    for _ in range(depth):
        layer = {w[:i] + w[i + 1:] for w in layer for i in range(len(w))}
        out |= layer
    return out


class PhraseMatcher:
    _MAX_CACHED_WORDS = 200_000

    def __init__(self, phrases: list[str], min_words: int):
        self.phrases = [words(p) for p in phrases if len(p.split()) >= min_words]
        self.vocabulary = sorted({w for p in self.phrases for w in p})
        # Deletion neighbourhoods: each edit is undone by at most one deletion
        # on each side, so a segment word within a phrase word's tolerance
        # shares a deletion string with it; only those candidates get the
        # exact distance.
        self._by_deletion: dict[str, set[str]] = {}
        for pw in self.vocabulary:
            for d in deletions(pw, tolerance(pw)):
                self._by_deletion.setdefault(d, set()).add(pw)
        self._longest = max(map(len, self.vocabulary))
        self._cache: dict[str, frozenset] = {}

    def _matches(self, word: str) -> frozenset:
        """Phrase words this segment word counts as (cached per word)."""
        hit = self._cache.get(word)
        if hit is not None:
            return hit
        if len(word) > self._longest + 2:  # more than 2 edits from every phrase word
            hit = frozenset()
        else:
            candidates = set().union(*(self._by_deletion.get(d, ()) for d in deletions(word, 2)))
            hit = frozenset(pw for pw in candidates if edits(pw, word) <= tolerance(pw))
        if len(self._cache) >= self._MAX_CACHED_WORDS:
            self._cache.clear()
        self._cache[word] = hit
        return hit

    @staticmethod
    def _aligned(phrase: list[str], window: list[frozenset]) -> int:
        """Most phrase words matched in order (longest common subsequence)."""
        prev = [0] * (len(window) + 1)
        for pw in phrase:
            cur = [0]
            for j, matches in enumerate(window):
                cur.append(prev[j] + 1 if pw in matches else max(prev[j + 1], cur[j]))
            prev = cur
        return prev[-1]

    def score(self, segment: str) -> float:
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

    def best(self, segments: list[str]) -> tuple[float, str | None]:
        """Highest score and its segment (0.0, None without segments)."""
        return max(((self.score(s), s) for s in segments), key=lambda x: x[0], default=(0.0, None))
