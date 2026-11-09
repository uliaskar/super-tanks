"""
eval_instruction.py
===================
Step 3 (../README.md): candidates for the instruction detector, which
recognises text that gives the assistant an order even without a known phrase.

Segment labels: a segment (a sentence of zef_segment.py) of an attacked tool output
is *injected* if it is part of the row's inserted text (column `payload`);
every other segment is not. A segment-level candidate learns from those labels
and scores a tool output by its highest segment score.

Candidates:
  C1        linear classifier on each segment's MiniLM embedding (step 2's model)
  C1-whole  linear classifier on the whole output's nomic-embed-text embedding
  C2a       linear classifier on word-list features
  C2b       C2a plus grammar features from spaCy (features_spacy.py)
  C2c       linear classifier on character fragments (TF-IDF)
Every candidate's threshold: BUDGET of the clean training outputs score above
it, using held-out scores (each training suite or benchmark scored by a model
trained without it).

Modes:
  --split selection   workspace <-> travel, both directions pooled, plus the
                      unseen-template test: each wrapped attack style left out
                      of training in turn and tested alone
  --split main        finalists: main split, alone and with the regex filter
  --robustness        finalists: robustness test, threshold from the main split
  --latency           scoring time per tool output
Every mode also reports ordinary Tatoeba sentences flagged.

Usage:
  python eval_instruction.py --split selection [--candidates ...]
"""

import argparse
import json
import re
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from core.security.zef_normalize import readable
from core.security.zef_segment import segments
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import eval_outputs as eo
import eval_phrase_similarity as ps
from embeddings import EMBED_ENV, HERE

BUDGET = 0.01
WRAPPED = ["station3_json", "station3_csv", "station3_html_comment", "station3_norwegian",
           "station3_russian", "station3_code_switch", "station3_zero_width"]
SPACY_CACHE = HERE / "data" / "feat_spacy.npz"


def squash(text):
    return re.sub(r"\W+", "", readable(text or "")).lower()


def segments_of(texts):
    return [segments(t) for t in texts]


def injected(segment, payload):
    """1 if the segment, with or without its `field: ` prefix, is part of the
    squashed payload (the field name is added by the segmenter)."""
    parts = [segment] + ([segment.split(": ", 1)[1]] if ": " in segment else [])
    return int(bool(payload) and any(len(squash(p)) >= 8 and squash(p) in payload for p in parts))


# ── Word-list features (C2a) ──────────────────────────────────────────────
_LISTS = {
    "second_person": r"\b(you|your|yours|du|deg|din|ditt|dine|dykk|ты|тебя|тебе|твой|твои|вы|вас)\b",
    "addressee": r"\b(assistant|ai|agent|model|bot|assistenten|ki|ассистент\w*|ии|бот\w*)\b",
    "urgency": r"\b(important|urgent|immediately|asap|priority|must|viktig\w*|haster|"
               r"umiddelbart|straks|срочно|важн\w*|немедленно)\b",
    "order": r"\b(first|before|then|after that|next|først|før|deretter|etterpå|сначала|"
             r"прежде|затем|потом)\b",
    "field_name": r"\b(instruction|instructions|priority|next_step|message_type|task|"
                  r"instruks\w*|oppgave\w*|инструкц\w*|задач\w*)\b",
    "action_verb": r"\b(send|transfer|pay|delete|forward|share|invite|book|reserve|buy|"
                   r"update|change|create|post|email|send\w*|overfør\w*|betal\w*|slett\w*|"
                   r"del|отправ\w*|переведи|удали\w*|оплат\w*)\b",
    "polite": r"\b(please|kindly|vennligst|vær så snill|пожалуйста)\b",
    "do_this": r"\b(do the following|do this|gjør dette|gjør følgande|сделай|выполни)\b",
}
_IDENT = r"([A-Z]{2}\d{2}[A-Z0-9]{8,30}|\S+@\S+\.\w+|https?://|www\.)"
_LIST_RES = {k: re.compile(v, re.IGNORECASE) for k, v in _LISTS.items()}


def lexical(segment):
    words = segment.split()
    return [float(bool(r.search(segment))) for r in _LIST_RES.values()] + [
        float(bool(re.search(_IDENT, segment))),
        float(segment.rstrip().endswith((":", "!"))),
        min(len(words), 40) / 40,
    ]


def spacy_features(segments):
    """C2b grammar features, cached in data/feat_spacy.npz."""
    cache = {}
    if SPACY_CACHE.exists():
        data = np.load(SPACY_CACHE)
        cache = dict(zip(data["keys"].tolist(), data["vectors"]))
    todo = sorted({s for s in segments if s not in cache})
    if todo:
        with tempfile.TemporaryDirectory() as tmp:
            sp, op = Path(tmp) / "segs.json", Path(tmp) / "out.npy"
            sp.write_text(json.dumps(todo, ensure_ascii=False), encoding="utf-8")
            subprocess.run([str(EMBED_ENV), str(HERE / "features_spacy.py"), str(sp), str(op)],
                           check=True)
            cache.update(zip(todo, np.load(op)))
        keys = list(cache)
        np.savez(SPACY_CACHE, keys=np.array(keys), vectors=np.stack([cache[k] for k in keys]))
    return np.stack([cache[s] for s in segments])


# ── Candidates ────────────────────────────────────────────────────────────
class SegmentCandidate:
    """A linear classifier on per-segment features; output score = max segment score."""

    def __init__(self, kind):
        self.kind = kind

    def features(self, segments, fit=False):
        if self.kind == "C1":
            return ps.embedder("paraphrase-multilingual-minilm")([s.lower() for s in segments])
        if self.kind == "C2c":
            low = [s.lower() for s in segments]
            if fit:
                self.tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5),
                                             min_df=2, sublinear_tf=True, max_features=200_000)
                return self.tfidf.fit_transform(low)
            return self.tfidf.transform(low)
        lex = np.array([lexical(s) for s in segments], dtype=np.float32)
        if self.kind == "C2b":
            return np.hstack([lex, spacy_features(segments)])
        return lex

    def fit(self, segments, labels):
        x = self.features(segments, fit=True)
        clf = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced")
        self.model = clf if sparse.issparse(x) else make_pipeline(StandardScaler(), clf)
        self.model.fit(x, labels)
        return self

    def score(self, segment_lists):
        unique = sorted({s for ss in segment_lists for s in ss})
        if not unique:
            return np.full(len(segment_lists), -1e9)
        best = dict(zip(unique, self.model.decision_function(self.features(unique))))
        return np.array([max((best[s] for s in ss), default=-1e9) for ss in segment_lists])


class WholeCandidate:
    """C1-whole: one nomic-embed-text embedding of the whole readable output."""
    MAX_CHARS = 6000

    def features(self, texts):
        return ps.embedder("nomic-embed-text")(
            [readable(t)[:self.MAX_CHARS].lower() for t in texts])

    def fit(self, texts, labels):
        self.model = make_pipeline(StandardScaler(), LogisticRegression(
            C=1.0, max_iter=3000, class_weight="balanced")).fit(self.features(texts), labels)
        return self

    def score(self, texts):
        return self.model.decision_function(self.features(texts))


CANDIDATES = ["C1", "C1-whole", "C2a", "C2b", "C2c"]


class Data:
    def __init__(self):
        self.rows = eo.load_rows()
        self.y = np.array([r["label"] for r in self.rows])
        self.framing = np.array([r["framing"] or "-" for r in self.rows])
        self.group = np.array([r["suite"] for r in self.rows])  # suite, or injecagent / bipia
        self.texts = [r["tool_output"] for r in self.rows]
        self.segs = segments_of(self.texts)
        payload = [squash(r["payload"]) if r["label"] else "" for r in self.rows]
        self.seg_labels = [[injected(s, p) for s in ss] for ss, p in zip(self.segs, payload)]

    def train_units(self, mask):
        """Distinct training segments; a segment is injected if it was injected anywhere."""
        label = {}
        for i in np.flatnonzero(mask):
            for s, lab in zip(self.segs[i], self.seg_labels[i]):
                label[s] = max(label.get(s, 0), lab)
        units = sorted(label)
        return units, np.array([label[u] for u in units])


def fit_candidate(name, data, mask):
    if name == "C1-whole":
        idx = np.flatnonzero(mask)
        return WholeCandidate().fit([data.texts[i] for i in idx], data.y[idx])
    units, labels = data.train_units(mask)
    return SegmentCandidate(name).fit(units, labels)


def score_rows(model, data, mask=None):
    idx = np.arange(len(data.rows)) if mask is None else np.flatnonzero(mask)
    out = np.full(len(data.rows), np.nan)
    if isinstance(model, WholeCandidate):
        out[idx] = model.score([data.texts[i] for i in idx])
    else:
        out[idx] = model.score([data.segs[i] for i in idx])
    return out


def held_out_threshold(name, data, train):
    """Threshold from held-out scores of the clean training outputs."""
    oof = np.full(len(data.rows), np.nan)
    for g in sorted(set(data.group[train])):
        inner = train & (data.group == g)
        model = fit_candidate(name, data, train & ~inner)
        oof[inner] = score_rows(model, data, inner)[inner]
    return float(np.quantile(oof[train & (data.y == 0)], 1 - BUDGET))


def tatoeba_flags(model, threshold, tatoeba):
    out = {}
    for lang, texts in tatoeba.items():
        if isinstance(model, WholeCandidate):
            s = model.score(texts)
        else:
            s = model.score(segments_of(texts))
        out[lang] = int((s > threshold).sum())
    return out


def selection(candidates):
    data = Data()
    tatoeba = ps.tatoeba_sentences()
    print(f"selection split, budget {BUDGET:.0%} of clean training outputs")
    print(f"segments: {sum(map(len, data.segs))}, injected: {sum(map(sum, data.seg_labels))}\n")
    folds = ps.selection_folds(data.rows)
    for name in candidates:
        caught = alarms = n_att = n_clean = 0
        aucs, styles, unseen = [], {w: [0, 0] for w in WRAPPED}, {w: [0, 0] for w in WRAPPED}
        tat = {lang: 0 for lang in tatoeba}
        for _, train, test in folds:
            thr = held_out_threshold(name, data, train)
            model = fit_candidate(name, data, train)
            s = score_rows(model, data, test)
            flagged = s > thr
            att, clean = test & (data.y == 1), test & (data.y == 0)
            caught += int(flagged[att].sum())
            n_att += int(att.sum())
            alarms += int(flagged[clean].sum())
            n_clean += int(clean.sum())
            aucs.append(roc_auc_score(data.y[test], s[test]))
            for w in WRAPPED:
                m = att & (data.framing == w)
                styles[w][0] += int(flagged[m].sum())
                styles[w][1] += int(m.sum())
            for lang, k in tatoeba_flags(model, thr, tatoeba).items():
                tat[lang] += k
            # unseen template: the style and its harmless controls left out of training
            for w in WRAPPED:
                keep = train & ~np.isin(data.framing, [w, f"control_{w}"])
                thr_w = held_out_threshold(name, data, keep)
                m = att & (data.framing == w)
                s_w = score_rows(fit_candidate(name, data, keep), data, m)
                unseen[w][0] += int((s_w[m] > thr_w).sum())
                unseen[w][1] += int(m.sum())
        print(f"{name:<9} attacks caught {caught:>5} of {n_att:<5} false alarms {alarms:>4} of "
              f"{n_clean:<5} AUC {np.mean(aucs):.3f} | Tatoeba flagged "
              + ", ".join(f"{lang} {k} of {2 * len(tatoeba[lang])}" for lang, k in tat.items()))
        print("   wrapped, trained with the style:    "
              + "  ".join(f"{w[9:]} {k}/{n}" for w, (k, n) in styles.items()))
        print("   wrapped, style unseen in training:  "
              + "  ".join(f"{w[9:]} {k}/{n}" for w, (k, n) in unseen.items()), flush=True)


def main_split(candidates):
    data = Data()
    train, test = eo.main_split(data.rows)
    regex = eo.regex_system(data.rows)
    tatoeba = ps.tatoeba_sentences()
    for name in candidates:
        thr = held_out_threshold(name, data, train)
        model = fit_candidate(name, data, train)
        flagged = score_rows(model, data) > thr
        eo.report(f"instruction-{name}", flagged, data.rows, test)
        eo.report(f"regex+instruction-{name}", flagged | regex, data.rows, test)
        for lang, k in tatoeba_flags(model, thr, tatoeba).items():
            print(f"  Tatoeba {lang}: {k} of {len(tatoeba[lang])} flagged")


class InstructionSystem:
    """A fitted candidate as a robustness-test system."""

    def __init__(self, model, threshold):
        self.model, self.threshold = model, threshold

    def prepare(self, outputs):
        if not isinstance(self.model, WholeCandidate):
            self.model.score([[s for ss in segments_of(outputs) for s in ss]])

    def __call__(self, outputs):
        if isinstance(self.model, WholeCandidate):
            return bool((self.model.score(outputs) > self.threshold).any())
        return bool((self.model.score(segments_of(outputs)) > self.threshold).any())


def robustness(candidates):
    import eval_robustness as er
    data = Data()
    train, _ = eo.main_split(data.rows)
    for name in candidates:
        thr = held_out_threshold(name, data, train)
        er.run(InstructionSystem(fit_candidate(name, data, train), thr), f"instruction-{name}")


def latency(candidates, n=200):
    data = Data()
    train, _ = eo.main_split(data.rows)
    sample = np.random.default_rng(0).choice(len(data.rows), n, replace=False)
    for name in candidates:
        model = fit_candidate(name, data, train)
        times = []
        for i in sample:
            text = data.texts[i]
            t = time.perf_counter()
            if isinstance(model, WholeCandidate):
                model.score([text])
            else:
                model.score(segments_of([text]))
            times.append((time.perf_counter() - t) * 1000)
        times.sort()
        print(f"{name:<9} ms per tool output (features cached): median {times[n // 2]:.0f}, "
              f"95th percentile {times[int(0.95 * (n - 1))]:.0f}")


def main(argv=None):
    p = argparse.ArgumentParser(description="Instruction-detector candidates (step 3).")
    p.add_argument("--split", choices=["selection", "main"])
    p.add_argument("--robustness", action="store_true")
    p.add_argument("--latency", action="store_true")
    p.add_argument("--candidates", nargs="+", choices=CANDIDATES, default=CANDIDATES)
    args = p.parse_args(argv)
    if args.split == "selection":
        selection(args.candidates)
    elif args.split == "main":
        main_split(args.candidates)
    if args.robustness:
        robustness(args.candidates)
    if args.latency:
        latency(args.candidates)


if __name__ == "__main__":
    main()
