"""
features_spacy.py
=================
Grammar features of text segments with spaCy, for the instruction detector's
C2b candidate (../README.md, step 3). Runs only inside harness/.venv-embed;
eval_instruction.py calls it.

Per segment (language picked from its letters: Cyrillic -> Russian, æ/ø/å or
common Norwegian words -> Bokmål, otherwise English):
  starts_with_verb   first word is a verb (or auxiliary)
  verb_no_subject    the main verb has no subject (typical for a command)
  imperative         some word has imperative mood
  second_person      some pronoun is in the second person
  verbs              number of verbs, capped at 5, divided by 5

Usage: .venv-embed/<python> features_spacy.py SEGMENTS.json OUT.npy
"""

import json
import re
import sys

import numpy as np
import spacy

MODELS = {"en": "en_core_web_sm", "nb": "nb_core_news_sm", "ru": "ru_core_news_sm"}
_NORWEGIAN = re.compile(r"[æøå]|\b(og|ikke|ikkje|jeg|eg|du|det|som|til|på|med|er|skal)\b",
                        re.IGNORECASE)


def language(text):
    if re.search(r"[Ѐ-ӿ]", text):
        return "ru"
    return "nb" if _NORWEGIAN.search(text) else "en"


def features(doc):
    tokens = [t for t in doc if not t.is_punct and not t.is_space]
    if not tokens:
        return [0.0] * 5
    root = next((t for t in doc if t.dep_ == "ROOT"), tokens[0])
    has_subject = any(c.dep_.startswith("nsubj") for c in root.children)
    return [
        float(tokens[0].pos_ in ("VERB", "AUX")),
        float(root.pos_ in ("VERB", "AUX") and not has_subject),
        float(any("Imp" in t.morph.get("Mood") for t in doc)),
        float(any(t.pos_ == "PRON" and "2" in t.morph.get("Person") for t in doc)),
        min(sum(t.pos_ in ("VERB", "AUX") for t in doc), 5) / 5,
    ]


def main():
    in_path, out_path = sys.argv[1:3]
    with open(in_path, encoding="utf-8") as f:
        segments = json.load(f)
    nlps = {lang: spacy.load(name, disable=["ner", "lemmatizer"]) for lang, name in MODELS.items()}
    out = np.zeros((len(segments), 5), dtype=np.float32)
    by_lang = {}
    for i, s in enumerate(segments):
        by_lang.setdefault(language(s), []).append(i)
    for lang, idx in by_lang.items():
        for i, doc in zip(idx, nlps[lang].pipe((segments[i] for i in idx), batch_size=256)):
            out[i] = features(doc)
    np.save(out_path, out)


if __name__ == "__main__":
    main()
