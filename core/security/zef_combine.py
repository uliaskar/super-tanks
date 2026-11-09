"""
core/security/zef_combine.py
=============================
ZEF detectors for tool outputs: runs the detectors on a tool result and
combines their scores with the regex verdict into one verdict.

Detectors (each scores the result's segments, zef_segment):
  phrase_similarity  rewordings / translations of known phrases (zef_phrase_similarity)
  phrase_match       known phrases despite misspellings (zef_phrase_match)
  instruction        text that gives the assistant an order (zef_instruction)
  hidden_order       the instruction score of segments that came from hidden
                     data (JSON, CSV, HTML comments inside a value)

Each detector has a flag threshold and a lower "medium" threshold
(zef_params/zef_params.json; None = never). The regex filter's own WARN or
BLOCK counts as one flag and one medium signal. Verdict: BLOCK if the regex
filter blocks or two flags agree; WARN on one flag or two medium signals;
else PASS. The verdict is never looser than the regex verdict. Thresholds and
rules were chosen in research/zef-v2 (eval_combine.py) at a false-alarm budget
of 1 % of clean training outputs.

On a WARN from the detectors, the result lists in `remove` every sentence over
the threshold of a detector that fired, and the sentence before and after each;
the gateway removes them (remove_sentences) and forwards the rest. Chosen in
the agent test (research/zef-v2/README.md): with the same verdicts, a notice above the output
left 116 of 2,241 wrapped attacks successful, removal 9.

Switch: ST_ZEF_DETECTORS=1 (off by default). Files: zef_params/ (in the
repository) and data/zef/ (scripts/zef_fetch_model.py), each checked against
the SHA-256 pinned here or in zef_phrase_similarity. If the switch is off, a
file is missing or does not match, a library is missing, or a detector
fails, the regex verdict stands and the reason is logged.
"""

import gzip
import hashlib
import json
import logging
import math
import os
import re
import threading
from pathlib import Path

from .zef_injection_filter import FilterResult, FilterVerdict
from .zef_normalize import readable
from .zef_segment import clauses, segment_origins

logger = logging.getLogger("zef.combine")

SWITCH = "ST_ZEF_DETECTORS"
PARAMS_DIR = Path(__file__).resolve().parent / "zef_params"
MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "zef"
PARAM_FILES = {
    "zef_params.json": "94440ee0dd668ba1ce567479ff0352267f9f90f17235a6cfd89cf635b913bc76",
    "zef_instruction.json.gz": "d3b835887ae6c9a0dd2b25de17b30465741658cbd852c303f8e847d8864a5abb",
}
# parameter-file key -> detector name in logs and matched_patterns
NAMES = {"B": "phrase_similarity", "A2": "phrase_match", "C2c": "instruction", "H": "hidden_order"}
# put in place of a removed sentence
REMOVED = "[removed by ZEF: suspected injected instruction]"

_RANK = {FilterVerdict.PASS: 0, FilterVerdict.WARN: 1, FilterVerdict.BLOCK: 2}
_lock = threading.Lock()
_loaded = False
_detectors = None


def enabled() -> bool:
    return os.getenv(SWITCH, "0").strip().lower() in ("1", "true", "yes")


def read_pinned(path: Path, sha256: str) -> bytes:
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != sha256:
        raise ValueError(f"{path} does not match its pinned SHA-256")
    return data


class Detectors:
    """The loaded detectors and thresholds."""

    def __init__(self, params_dir: Path | None = None, model_dir: Path | None = None):
        from .zef_instruction import InstructionDetector
        from .zef_phrase_match import PhraseMatcher
        from .zef_phrase_similarity import MODEL_FILES, PhraseSimilarity

        params_dir, model_dir = params_dir or PARAMS_DIR, model_dir or MODEL_DIR
        params = json.loads(read_pinned(params_dir / "zef_params.json", PARAM_FILES["zef_params.json"]))
        instruction = json.loads(gzip.decompress(read_pinned(
            params_dir / "zef_instruction.json.gz", PARAM_FILES["zef_instruction.json.gz"])))
        for name, (_, sha256) in MODEL_FILES.items():
            read_pinned(model_dir / name, sha256)
        self.thresholds = {k: tuple(math.inf if t[x] is None else t[x] for x in ("flag", "medium"))
                           for k, t in params["thresholds"].items()}
        self.similarity = PhraseSimilarity(model_dir, params["phrases"])
        self.matcher = PhraseMatcher(params["phrases"], params["phrase_match_min_words"])
        self.instruction = InstructionDetector(instruction)

    def scores(self, result) -> dict:
        """Detector key -> (score, segment it came from)."""
        origins = segment_origins(result)
        sentences = [s for s, _ in origins]
        hidden = [s for s, h in origins if h]
        with_clauses = list(dict.fromkeys(sentences + [c for s in sentences for c in clauses(s)]))
        return {"B": self.similarity.best(with_clauses), "A2": self.matcher.best(with_clauses),
                "C2c": self.instruction.best(sentences), "H": self.instruction.best(hidden)}

    def to_remove(self, result, over: dict) -> list[str]:
        """Sentences scoring above `over` (detector key -> threshold) for any
        detector, scored as scores() does, and the sentence before and after each."""
        from .zef_instruction import NO_SEGMENT

        origins = segment_origins(result)
        parts = [[s, *clauses(s)] for s, _ in origins]
        similarity = iter(self.similarity.each([p for ps in parts for p in ps]).tolist())
        hit = set()
        for j, ((s, hidden), ps) in enumerate(zip(origins, parts)):
            c = self.instruction.score(s)
            score = {"B": max(next(similarity) for _ in ps), "A2": self.matcher.best(ps)[0],
                     "C2c": c, "H": c if hidden else NO_SEGMENT}
            if any(score[k] > t for k, t in over.items()):
                hit.update((j - 1, j, j + 1))
        return [origins[j][0] for j in sorted(hit) if 0 <= j < len(origins)]


def _patterns(sentence: str) -> list:
    """The sentence's words with any spacing, quoting or escapes between them,
    as the segmenter's joined text may differ from the raw value; with and
    without its "field: " prefix."""
    out = []
    for candidate in (sentence, sentence.split(": ", 1)[-1]):
        words = re.findall(r"\w+", candidate)
        if words:
            out.append(re.compile(r"[\W_]*".join(map(re.escape, words)), re.IGNORECASE))
    return out


def _map_text(value, fn):
    if isinstance(value, str):
        return fn(value)
    if isinstance(value, dict):
        return {k: _map_text(v, fn) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_map_text(v, fn) for v in value)
    return value


def remove_sentences(result, sentences: list[str]):
    """The result, same structure, with every occurrence of each sentence in
    its text values replaced by REMOVED (a value is searched as written, then
    decoded with readable()); None if a sentence is not found."""
    for sentence in sentences:
        for pattern in _patterns(sentence):
            found = False

            def cut(text, pattern=pattern):
                nonlocal found
                for form in (text, readable(text)):
                    new, n = pattern.subn(REMOVED, form)
                    if n:
                        found = True
                        return new
                return text

            cleaned = _map_text(result, cut)
            if found:
                result = cleaned
                break
        else:
            return None
    return result


def combine(regex: FilterVerdict, scores: dict, thresholds: dict) -> tuple[FilterVerdict, list, list]:
    """(verdict, keys of detectors that flag, keys that give a medium signal)."""
    flags = [k for k, (s, _) in scores.items() if s > thresholds[k][0]]
    medium = [k for k, (s, _) in scores.items() if s > thresholds[k][1]]
    by_regex = int(regex is not FilterVerdict.PASS)
    if regex is FilterVerdict.BLOCK or by_regex + len(flags) >= 2:
        verdict = FilterVerdict.BLOCK
    elif by_regex + len(flags) >= 1 or by_regex + len(medium) >= 2:
        verdict = FilterVerdict.WARN
    else:
        verdict = FilterVerdict.PASS
    return max(verdict, regex, key=_RANK.get), flags, medium


def _load():
    """The detectors, loaded once; None if they cannot be (reason logged once)."""
    global _loaded, _detectors
    with _lock:
        if not _loaded:
            _loaded = True
            try:
                _detectors = Detectors()
                logger.info("[ZEF] detectors loaded")
            except Exception as e:  # missing file, checksum, library: regex verdict stands
                logger.error("[ZEF] detectors unavailable, regex filter only: %s", e)
        return _detectors


def scan_tool_result(result, regex: FilterResult) -> FilterResult:
    """The regex result, made stricter if the detectors flag the tool result."""
    if not enabled():
        return regex
    detectors = _load()
    if detectors is None:
        return regex
    try:
        scores = detectors.scores(result)
    except Exception as e:  # a failing detector gives no flags
        logger.error("[ZEF] detectors failed, regex verdict stands: %s", e)
        return regex
    verdict, flags, medium = combine(regex.verdict, scores, detectors.thresholds)
    if verdict is FilterVerdict.PASS or not (flags or medium):
        return regex
    fired = flags or [k for k in medium if k not in flags]
    patterns = [f"zef:{NAMES[k]}" + ("" if k in flags else ":medium") for k in fired]
    segments = list(dict.fromkeys(scores[k][1] for k in fired if scores[k][1]))
    remove = []
    if verdict is FilterVerdict.WARN:
        over = {k: detectors.thresholds[k][0 if k in flags else 1] for k in dict.fromkeys(flags + medium)}
        try:
            remove = detectors.to_remove(result, over)
        except Exception as e:
            logger.error("[ZEF] could not locate the flagged sentences: %s", e)
        if not remove:  # flagged text that cannot be removed is not forwarded
            verdict = FilterVerdict.BLOCK
    return FilterResult(
        verdict=verdict,
        message=f"ZEF detectors: {', '.join(patterns)}",
        matched_patterns=list(regex.matched_patterns) + patterns,
        flagged_segments=segments,
        remove=remove,
    )
