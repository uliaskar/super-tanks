"""
core/security/zef_normalize.py
===============================
The ZEF normalizer: undoes the tricks that hide an injection from the regex
filter and the detectors.

Two forms of a text:

* `canonical()` folds every non-ASCII look-alike (Cyrillic / Greek / Armenian /
  full-width / math alphanumerics) onto ASCII, so homoglyph-spoofed keywords
  match the ASCII regex catalogue. FOR MATCHING ONLY: characters outside the
  confusables table are left as-is, so the result can mix scripts. Never
  display or log it.
* `readable()` decodes escapes and encodings (serializer escapes, URL
  encoding, HTML entities, readable base64) and folds look-alikes only inside
  words that mix scripts, so real Cyrillic or Norwegian text stays intact.
  Input for the detectors; the regex filter also matches
  `canonical(readable(text))` next to the raw text.

Data source: the ``confusables`` package (MIT), which ships the Unicode
confusables table and an alpha-prioritised ``normalize()``. Pure Python,
offline. Pinned: the table grows with each Unicode release.

Only NON-ASCII characters are folded. ASCII is never rewritten, so ASCII
punctuation the regex catalogue relies on (back-ticks, brackets, slashes) is
preserved; the library would otherwise map some ASCII to other ASCII
look-alikes (back-tick to apostrophe).

Known gap: a few code points have no ASCII mapping in the table (e.g.
Cherokee U+13C1, U+13BE), so a word built entirely from those does not fold.
"""

import base64
import html
import re
import unicodedata
import urllib.parse
from functools import lru_cache

from confusables import normalize as _confusable_forms

# Ceiling on candidate forms scanned when folding a single non-ASCII char.
_MAX_FORMS = 8


def _strip_format_chars(text: str) -> str:
    """Drop Unicode category Cf (zero-width space/joiner/non-joiner, BOM, ...),
    which an attacker inserts to break a regex word boundary inside a keyword."""
    return "".join(ch for ch in text if unicodedata.category(ch) != "Cf")


@lru_cache(maxsize=16384)
def _fold_char(ch: str) -> str:
    """Best ASCII look-alike for a single non-ASCII character, or the character
    unchanged if the confusables table has no ASCII mapping for it."""
    try:
        forms = _confusable_forms(ch, prioritize_alpha=True) or []
    except Exception:
        return ch
    ascii_forms = [f for f in forms[:_MAX_FORMS] if f and all(ord(c) < 128 for c in f)]
    if not ascii_forms:
        return ch
    # Shortest ASCII rendering keeps 1:1 look-alikes 1 char (avoids expansions).
    return min(ascii_forms, key=len)


@lru_cache(maxsize=8192)
def canonical(text: str) -> str:
    """Return a single ASCII-directed canonical form of *text* for matching.

    Passes:
      1. Strip Cf (invisible) characters.
      2. NFKC - fold compatibility variants (full-width, ligatures, math
         alphanumerics, superscripts) to canonical form.
      3. Per-character confusables fold: each NON-ASCII character is mapped to
         its best ASCII look-alike; ASCII characters are left untouched.

    Combining marks are preserved, so precomposed accented letters (Norwegian
    a-ring, o-slash) are not damaged. NOT for display - see module docstring.
    """
    if not text:
        return text

    base = unicodedata.normalize("NFKC", _strip_format_chars(text))
    out = []
    for ch in base:
        out.append(ch if ord(ch) < 128 else _fold_char(ch))
    return "".join(out)


# Escapes a YAML/JSON serializer writes for non-ASCII and control characters
# (\xF8, \u0412, \U0001D5C2), a YAML line fold (backslash, newline, indent,
# optional escaped space) and the simple escapes \n \t \" \\.
_ESCAPE = re.compile(
    r'\\(x[0-9A-Fa-f]{2}|u[0-9A-Fa-f]{4}|U[0-9A-Fa-f]{8}|\n[ \t]*(?:\\(?= ))?|[nt"\\])'
)
_SIMPLE_ESCAPES = {"n": "\n", "t": "\t", '"': '"', "\\": "\\"}


def _unescape(match: "re.Match[str]") -> str:
    code = match.group(1)
    if code[0] in "xuU":
        cp = int(code[1:], 16)
        # Surrogates and out-of-range values are not characters; keep as written.
        return match.group(0) if 0xD800 <= cp <= 0xDFFF or cp > 0x10FFFF else chr(cp)
    if code[0] == "\n":
        return ""
    return _SIMPLE_ESCAPES[code]


# Encodings an attacker can wrap an instruction in that a capable LLM still
# reads: URL percent-encoding, HTML entities, base64. Decoded in a bounded loop
# so double encoding (base64 of URL-encoded text, escaped twice) unwraps too.
MAX_DECODE_PASSES = 3
_URL_ESCAPE = re.compile(r"%[0-9A-Fa-f]{2}")
_HTML_ENTITY = re.compile(r"&(?:#\d+|#[xX][0-9A-Fa-f]+|[A-Za-z]+);")
_BASE64_BLOCK = re.compile(r"(?<![A-Za-z0-9+/=])[A-Za-z0-9+/]{24,}={0,2}(?![A-Za-z0-9+/=])")
_MAX_BASE64_BLOCK = 20_000


def _decode_base64(match: "re.Match[str]") -> str:
    """Decode a base64 block only if it is readable prose. IDs, hashes, keys and
    binary data stay as they are: they fail UTF-8 decoding or the
    printable / letters / spaces checks."""
    block = match.group(0)
    if len(block) > _MAX_BASE64_BLOCK:
        return block
    padded = block if "=" in block else block + "=" * (-len(block) % 4)
    try:
        text = base64.b64decode(padded, validate=True).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return block
    # Judge readability with the inner URL/HTML layers undone (double encoding),
    # but return the base64-decoded text; the next pass unwraps the rest.
    view = html.unescape(urllib.parse.unquote(text))
    printable = sum(ch.isprintable() or ch in "\n\t" for ch in view)
    letters = sum(ch.isalpha() for ch in view)
    if printable < 0.95 * len(view) or letters < 0.5 * len(view) or " " not in view:
        return block
    return text


def _decode_layers(text: str) -> str:
    """Serializer escapes, URL encoding, HTML entities and readable base64,
    repeated until nothing changes (at most MAX_DECODE_PASSES)."""
    for _ in range(MAX_DECODE_PASSES):
        before = text
        text = _ESCAPE.sub(_unescape, text)
        if _URL_ESCAPE.search(text):
            text = urllib.parse.unquote(text)
        if _HTML_ENTITY.search(text):
            text = html.unescape(text)
        text = _BASE64_BLOCK.sub(_decode_base64, text)
        if text == before:
            break
    return text


def readable(text: str) -> str:
    """Return *text* as plain readable Unicode, for the detectors and as the
    regex filter's second form.

    Tool results serialized as YAML (AgentDojo) or JSON arrive with non-ASCII
    written as escapes, so Norwegian, Cyrillic or a zero-width space appear as
    "\\xF8", "\\u0412", "\\u200B"; attackers add URL encoding, HTML entities or
    base64.

    Passes:
      1. Decode layers (see _decode_layers): serializer escapes and YAML line
         folds (any script), URL encoding, HTML entities, readable base64,
         repeated until nothing changes.
      2. Strip Cf (invisible) characters - same rule as canonical().
      3. NFKC - same rule as canonical().
      4. Fold look-alikes only inside mixed-script words (see _fold_mixed).

    Unlike canonical(), a word written wholly in another script is NOT folded
    to ASCII: that would turn real Cyrillic text into Latin noise.
    """
    if not text:
        return text
    text = unicodedata.normalize("NFKC", _strip_format_chars(_decode_layers(text)))
    return _WORD.sub(_fold_mixed, text)


_WORD = re.compile(r"\w+")


@lru_cache(maxsize=4096)
def _is_latin(ch: str) -> bool:
    return unicodedata.name(ch, "").startswith("LATIN")


def _fold_mixed(match: "re.Match[str]") -> str:
    """Fold the non-Latin letters of a word that mixes scripts.

    A word with Latin letters plus letters of another script is a homoglyph
    disguise, e.g. ignore spelled with Cyrillic U+0456 and U+043E. Its
    non-Latin letters are mapped to their ASCII look-alike. A word in a single
    script is left alone: Cyrillic words stay Cyrillic, and Norwegian words
    keep their o-slash and a-ring (those are Latin letters).
    """
    word = match.group(0)
    letters = [c for c in word if c.isalpha()]
    if not any(_is_latin(c) for c in letters) or all(_is_latin(c) for c in letters):
        return word
    return "".join(c if not c.isalpha() or _is_latin(c) else _fold_char(c) for c in word)
