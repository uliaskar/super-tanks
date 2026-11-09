"""
core/security/zef_segment.py
=============================
The ZEF segmenter: cuts a tool result into the segments the detectors judge.

A tool result arrives as an object (dict / list / str, as the tool returned
it) or as text that serializes one (JSON, YAML, a Python literal). Structure
is read first, so a field's value stays whole: YAML wraps long strings over
several lines, and cutting at those line breaks splits an injected sentence
into harmless-looking pieces.

* Every text value becomes `key: sentence` segments, under its innermost field
  name (`subject: Before you can solve ...`); field names such as
  `instruction` or `priority` are signals for the detectors.
* A value that itself holds JSON, YAML, a Python literal, CSV with a header
  row or HTML comments is read the same way, at most MAX_DEPTH levels deep;
  this is where wrapped injections hide. JSON objects inside plain text (a
  file with a JSON line in the middle) are cut out and read the same way.
* Text is passed through `readable()` (zef_normalize) and cut into sentences:
  at sentence-ending punctuation, blank lines and lines that start a list
  item or a `key:` field. A plain line break inside a paragraph is joined.

`segment_origins()` also says which segments came from hidden data: JSON, a
Python literal or CSV inside a text value, or JSON objects and HTML comments
inside text, but not the tool result's own outer structure. A value that only
reads as YAML (`Purchase at Apple Store: iPhone 3GS`) is not hidden data;
ordinary values look like that. Real tool results rarely carry an order
inside hidden data, wrapped injections do.

`clauses()` cuts a sentence further, for detectors that compare segments with
short phrases. There is no segment cap: an attacker would put the injection
after it. Callers bound cost by batching; repeated segments are dropped here.

Parsing is bounded: texts above MAX_PARSE_CHARS are treated as plain text, YAML
with anchors is not parsed (alias expansion), at most MAX_EMBEDDED_TRIES `{`
positions per text are tried as embedded JSON, and any parser error falls back
to plain text, so a malformed or hostile structure is still segmented.
"""

import ast
import csv
import io
import json
import re

import yaml

from .zef_normalize import readable

MAX_DEPTH = 3
MAX_PARSE_CHARS = 100_000
MAX_CLAUSE_WORDS = 40
MAX_EMBEDDED_TRIES = 64

_JSON = json.JSONDecoder()
_YAML_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
_YAML_ANCHOR = re.compile(r"(?:^|[\s\[{,])&[^\s]")
_YAML_START = re.compile(r"\s*(?:-\s|[\w\"' -]{1,40}:(?:\s|$))")
_SENTENCE = re.compile(
    r"(?<=[.!?])\s+"                                    # sentence end
    r"|\n\s*\n+"                                        # blank line
    r"|\n(?=\s*(?:[-*•]\s|\d+[.)]\s|\w{1,30}:\s))"  # list item or key: field
)
_CLAUSE = re.compile(r"\s*(?:[,;:]|\s-\s|\s(?:and|og|then|så)\s)\s*", re.IGNORECASE)
_HTML_COMMENT = re.compile(r"<!--(.*?)-->", re.DOTALL)
_CSV_HEADER = re.compile(r"[A-Za-z_ ]{2,30}")


def segments(result) -> list[str]:
    """Segments of a tool result (object or text), in order, without repeats."""
    return [s for s, _ in segment_origins(result)]


def segment_origins(result) -> list[tuple[str, bool]]:
    """(segment, came from hidden data) per segment, in order, without repeats;
    a repeated segment counts as hidden if any occurrence is."""
    out: list[tuple[str, bool]] = []
    # an object result is already parsed: its values are one level down, as a
    # text result's values are after parsing
    _walk(result, "", 0 if isinstance(result, (str, bytes)) else 1, False, out)
    hidden: dict[str, bool] = {}
    for s, h in out:
        hidden[s] = hidden.get(s, False) or h
    return list(hidden.items())


def clauses(sentence: str) -> list[str]:
    """Parts of a sentence between commas, semicolons, colons, dashes and
    "and / og / then / så", each at least 2 words; longer parts in windows of
    MAX_CLAUSE_WORDS words."""
    out = []
    for part in _CLAUSE.split(sentence):
        words = part.split()
        if len(words) < 2:
            continue
        for i in range(0, len(words), MAX_CLAUSE_WORDS):
            window = words[i:i + MAX_CLAUSE_WORDS]
            if len(window) >= 2:
                out.append(" ".join(window))
    return out


def _walk(value, key: str, depth: int, hidden: bool, out: list) -> None:
    """Text values of a parsed object, depth first, in order. Iterative, so
    deeply nested objects cannot exhaust the call stack."""
    stack = [(value, key)]
    while stack:
        value, key = stack.pop()
        if isinstance(value, dict):
            stack.extend((v, str(k)) for k, v in reversed(list(value.items())))
        elif isinstance(value, (list, tuple)):
            stack.extend((v, key) for v in reversed(value))
        elif isinstance(value, str):
            _text(value, key, depth, hidden, out)
        elif isinstance(value, bytes):
            _text(value.decode("utf-8", "replace"), key, depth, hidden, out)
        # numbers, booleans and None carry no instruction


def _text(text: str, key: str, depth: int, hidden: bool, out: list) -> None:
    if not text.strip():
        return
    plain = readable(text)
    if depth < MAX_DEPTH:
        obj = _parse(text)
        if obj is None and plain != text:
            obj = _parse(plain)
        csv_rows = obj is None
        if csv_rows:
            obj = _csv(plain)
        if obj is not None:
            # depth 0 is the tool result's own structure; inside a value, JSON,
            # a Python literal or CSV is hidden data, plain YAML is not
            inner = depth > 0 and (csv_rows or plain.lstrip()[:1] in ("{", "["))
            _walk(obj, key, depth + 1, hidden or inner, out)
            return
        objs, plain = _embedded_json(plain)
        for obj in objs:
            _walk(obj, key, depth + 1, True, out)
        for comment in _HTML_COMMENT.findall(plain):
            _text(comment, "comment", depth + 1, True, out)
    plain = _HTML_COMMENT.sub(" ", plain)
    for sentence in _SENTENCE.split(plain):
        words = sentence.split()
        if len(words) >= 2:
            out.append((f"{key}: {' '.join(words)}" if key else " ".join(words), hidden))


def _parse(text: str):
    """A dict or list serialized as JSON, a Python literal or YAML, else None."""
    t = text.strip()
    if not t or len(t) > MAX_PARSE_CHARS:
        return None
    loaders = []
    if t[0] in "{[":
        loaders += [json.loads, ast.literal_eval]
    if (t[0] in "{[" or _YAML_START.match(t)) and not _YAML_ANCHOR.search(t):
        loaders.append(lambda s: yaml.load(s, Loader=_YAML_LOADER))  # noqa: S506 (safe loader)
    for load in loaders:
        try:
            obj = load(t)
        except Exception:  # any malformed or hostile input falls back to text
            continue
        if isinstance(obj, (dict, list)):
            return obj
    return None


def _embedded_json(text: str):
    """JSON objects inside plain text, and the text with each object replaced
    by a line break."""
    if "{" not in text or len(text) > MAX_PARSE_CHARS:
        return [], text
    objs, kept, pos = [], [], 0
    start = text.find("{")
    for _ in range(MAX_EMBEDDED_TRIES):
        if start == -1:
            break
        try:
            obj, end = _JSON.raw_decode(text, start)
        except ValueError:
            obj = None
        if isinstance(obj, dict) and obj:
            objs.append(obj)
            kept.append(text[pos:start])
            pos = end
            start = text.find("{", end)
        else:
            start = text.find("{", start + 1)
    kept.append(text[pos:])
    return objs, "\n\n".join(kept)


def _csv(text: str):
    """CSV with a header row of field names, as a list of {header: cell} rows."""
    if "," not in text or len(text) > MAX_PARSE_CHARS:
        return None
    try:
        rows = [r for r in csv.reader(io.StringIO(text.strip())) if r]
    except csv.Error:
        return None
    if len(rows) < 2 or len(rows[0]) < 2 or any(len(r) != len(rows[0]) for r in rows):
        return None
    head = [h.strip() for h in rows[0]]
    if not all(_CSV_HEADER.fullmatch(h) for h in head):
        return None
    return [dict(zip(head, r)) for r in rows[1:]]
