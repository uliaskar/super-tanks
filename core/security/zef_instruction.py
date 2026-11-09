"""
core/security/zef_instruction.py
=================================
ZEF instruction detector: how much does a segment read like an order to the
assistant?

Character fragments: every 2- to 5-letter piece of each word (the word padded
with a space on both sides), weighted by how rare the piece is (idf), then a
linear model. This is scikit-learn's TfidfVectorizer(analyzer="char_wb",
ngram_range=(2, 5), sublinear_tf=True, norm="l2") followed by
LogisticRegression.decision_function, reimplemented in plain Python so the
runtime needs no scikit-learn. Trained in research/zef-v2
(eval_instruction.py); the parameters are loaded from zef_params/zef_instruction.json.gz.
"""

import math
import re
from collections import Counter

NO_SEGMENT = -1e9  # score when there is nothing to judge

_WHITE_SPACES = re.compile(r"\s\s+")


def fragments(text: str, low: int, high: int) -> list[str]:
    """Character n-grams of each word, as scikit-learn's char_wb analyzer."""
    out = []
    for w in _WHITE_SPACES.sub(" ", text.lower()).split():
        w = f" {w} "
        for n in range(low, high + 1):
            offset = 0
            out.append(w[offset:offset + n])
            while offset + n < len(w):
                offset += 1
                out.append(w[offset:offset + n])
            if offset == 0:  # the word is shorter than n: longer n-grams are the same
                break
    return out


class InstructionDetector:
    def __init__(self, params: dict):
        self.low, self.high = params["ngram_range"]
        self.intercept = params["intercept"]
        self.weights = {g: (idf, coef) for g, idf, coef in params["features"]}

    def score(self, segment: str) -> float:
        dot = norm = 0.0
        for g, count in Counter(fragments(segment, self.low, self.high)).items():
            w = self.weights.get(g)
            if w is None:
                continue
            x = (1.0 + math.log(count)) * w[0]
            dot += x * w[1]
            norm += x * x
        return dot / math.sqrt(norm) + self.intercept if norm else self.intercept

    def best(self, segments: list[str]) -> tuple[float, str | None]:
        """Highest score and its segment (NO_SEGMENT, None without segments)."""
        return max(((self.score(s), s) for s in segments), key=lambda x: x[0],
                   default=(NO_SEGMENT, None))
