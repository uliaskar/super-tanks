"""
eval_phrase_similarity.py
=========================
Step 2 (../README.md): candidates for phrase similarity, i.e. recognising
rewordings and translations of known injection phrases.

A candidate = a phrase list (phrase_list.py) + an embedding model
(embeddings.py). A tool output's score is the highest cosine similarity
between any of its segments and any phrase; it is flagged above a threshold
set so that BUDGET of the clean training outputs score above it. Segments:
the sentences of Super Tanks' segmenter (zef_segment.py) plus their clauses,
since the phrases are short.

Modes:
  --split selection   choose candidates: train on workspace (+ external
                      training rows), test on travel, then the reverse
  --split main        finalists only: main split, alone and with the regex filter
  --robustness        finalists only: robustness test, threshold from the main split
  --latency           embedding time per tool output, model already loaded
Every mode also reports the share of ordinary Russian, Bokmål and Nynorsk
sentences (Tatoeba) flagged at the same threshold.

Usage:
  python eval_phrase_similarity.py --split selection [--candidates ...]
  python eval_phrase_similarity.py --split main --candidates B1-nomic B2-bge-m3
"""

import argparse
import bz2
import functools
import json
import random
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from core.security import zef_segment
from sklearn.metrics import roc_auc_score

import eval_outputs as eo
import phrase_list
from embeddings import BATCH, EMBED_ENV, HERE, MODELS, Embedder, _ollama

BUDGET = 0.01
TATOEBA = HERE / "data" / "external" / "tatoeba"
TATOEBA_PER_LANGUAGE = 1000

CANDIDATES = {"reference": (phrase_list.VERSION_1, "nomic-embed-text")}
CANDIDATES.update({f"B1-{m}" if m == "nomic-embed-text" else f"B2-{m}": (phrase_list.expanded(), m)
                   for m in MODELS})


def tatoeba_sentences():
    """A fixed sample of ordinary sentences per language (3+ words)."""
    out = {}
    for lang in ("rus", "nob", "nno"):
        with bz2.open(TATOEBA / f"{lang}_sentences.tsv.bz2", "rt", encoding="utf-8") as f:
            texts = [line.rstrip("\n").split("\t")[2] for line in f]
        texts = sorted(t for t in texts if len(t.split()) >= 3)
        out[lang] = random.Random(0).sample(texts, min(TATOEBA_PER_LANGUAGE, len(texts)))
    return out


def segments_of(texts):
    """Per text: its sentences and their clauses, without repeats."""
    out = []
    for t in texts:
        sentences = zef_segment.segments(t)
        out.append(list(dict.fromkeys(
            sentences + [c for s in sentences for c in zef_segment.clauses(s)])))
    return out


@functools.cache
def embedder(model):
    """One cached Embedder per model per run (loading a cache file is slow)."""
    return Embedder(model)


def scores(candidate, segment_lists):
    """Highest segment-to-phrase cosine per text (-1 if a text has no segment).
    Phrases and segments are lowercased, so letter case plays no role (the
    normalizer's readable() form keeps case)."""
    phrases, model = CANDIDATES[candidate]
    embed = embedder(model)
    phrase_vecs = embed([p.lower() for p in phrases])
    segment_lists = [[s.lower() for s in ss] for ss in segment_lists]
    unique = sorted({s for ss in segment_lists for s in ss})
    vecs = embed(unique) if unique else np.zeros((0, phrase_vecs.shape[1]))
    best = dict(zip(unique, (vecs @ phrase_vecs.T).max(axis=1))) if unique else {}
    return np.array([max((best[s] for s in ss), default=-1.0) for ss in segment_lists])


def selection_folds(rows):
    part = eo._external_part(rows)
    ext_train = np.array([p == "train" for p in part])
    suite = np.array([r["suite"] if r["benchmark"] == "agentdojo" else "" for r in rows])
    return [("train workspace, test travel", (suite == "workspace") | ext_train, suite == "travel"),
            ("train travel, test workspace", (suite == "travel") | ext_train, suite == "workspace")]


def _count(flags, mask):
    return int((flags & mask).sum()), int(mask.sum())


def selection(candidates, rows):
    y = np.array([r["label"] for r in rows])
    framing = np.array([r["framing"] or "-" for r in rows])
    seg_rows = segments_of([r["tool_output"] for r in rows])
    tatoeba = tatoeba_sentences()
    seg_tat = {lang: segments_of(texts) for lang, texts in tatoeba.items()}
    styles = ["override_en_synonym", "override_nb_synonym", "override_en_typo",
              "station3_norwegian", "station3_russian", "station3_code_switch",
              "station3_json", "gen_before_en", "gen_before_nb"]
    print(f"selection split, budget {BUDGET:.0%} of clean training outputs\n")
    for name in candidates:
        s = scores(name, seg_rows)
        tat = {lang: scores(name, ss) for lang, ss in seg_tat.items()}
        caught = alarms = n_att = n_clean = 0
        tat_flagged = {lang: 0 for lang in tat}
        style_counts = {st: [0, 0] for st in styles}
        aucs = []
        for _, train, test in selection_folds(rows):
            thr = eo.threshold_at_budget(s, train & (y == 0), BUDGET)
            flagged = s > thr
            k, n = _count(flagged, test & (y == 1))
            caught, n_att = caught + k, n_att + n
            k, n = _count(flagged, test & (y == 0))
            alarms, n_clean = alarms + k, n_clean + n
            aucs.append(roc_auc_score(y[test], s[test]))
            for lang, t in tat.items():
                tat_flagged[lang] += int((t > thr).sum())
            for st in styles:
                k, n = _count(flagged, test & (y == 1) & (framing == st))
                style_counts[st][0] += k
                style_counts[st][1] += n
        tat_text = ", ".join(f"{lang} {k} of {2 * len(tatoeba[lang])}"
                             for lang, k in tat_flagged.items())
        print(f"{name:<36} attacks caught {caught:>5} of {n_att:<5} false alarms "
              f"{alarms:>4} of {n_clean:<5} AUC {np.mean(aucs):.3f} | Tatoeba flagged: {tat_text}")
        print("    " + "  ".join(f"{st} {k}/{n}" for st, (k, n) in style_counts.items() if n))


def main_split(candidates, rows):
    y = np.array([r["label"] for r in rows])
    train, test = eo.main_split(rows)
    seg_rows = segments_of([r["tool_output"] for r in rows])
    regex = eo.regex_system(rows)
    tatoeba = tatoeba_sentences()
    for name in candidates:
        s = scores(name, seg_rows)
        thr = eo.threshold_at_budget(s, train & (y == 0), BUDGET)
        eo.report(name, s > thr, rows, test)
        eo.report(f"regex+{name}", (s > thr) | regex, rows, test)
        for lang, texts in tatoeba.items():
            t = scores(name, segments_of(texts))
            print(f"  Tatoeba {lang}: {int((t > thr).sum())} of {len(texts)} flagged")


def robustness(candidates, rows):
    import eval_robustness as er
    y = np.array([r["label"] for r in rows])
    train, _ = eo.main_split(rows)
    seg_rows = segments_of([r["tool_output"] for r in rows])
    for name in candidates:
        phrases, _ = CANDIDATES[name]
        er.check_held_out(phrases)
        thr = eo.threshold_at_budget(scores(name, seg_rows), train & (y == 0), BUDGET)
        er.run(PhraseSystem(name, thr), f"phrase-{name}")


class PhraseSystem:
    """A candidate as a robustness-test system: caught if any tool output
    scores above the threshold."""

    def __init__(self, name, threshold):
        self.name, self.threshold = name, threshold

    def prepare(self, outputs):  # embed every segment of the test in one batch
        scores(self.name, [[s for ss in segments_of(outputs) for s in ss]])

    def __call__(self, outputs):
        return bool((scores(self.name, segments_of(outputs)) > self.threshold).any())


def latency(candidates, rows, n=200):
    sample = random.Random(0).sample([r["tool_output"] for r in rows], n)
    groups = segments_of(sample)
    def q(values, p):
        return sorted(values)[int(p * (len(values) - 1))]

    for name in candidates:
        backend, model_id, prefix = MODELS[CANDIDATES[name][1]]
        groups_p = [[prefix + s for s in g] or ["empty"] for g in groups]
        if backend == "ollama":
            # time this model alone: unload every other Ollama model first
            for other in {m[1] for m in MODELS.values() if m[0] == "ollama"} - {model_id}:
                subprocess.run(["ollama", "stop", other], capture_output=True, check=False)
            _ollama(model_id, ["warm-up"])
            times = []
            for g in groups_p:
                t = time.perf_counter()
                for i in range(0, len(g), BATCH):  # Ollama fails on 256+ texts per request
                    _ollama(model_id, g[i:i + BATCH])
                times.append((time.perf_counter() - t) * 1000)
        else:
            with tempfile.TemporaryDirectory() as tmp:
                gp, op = Path(tmp) / "groups.json", Path(tmp) / "times.json"
                gp.write_text(json.dumps(groups_p, ensure_ascii=False), encoding="utf-8")
                subprocess.run([str(EMBED_ENV), str(HERE / "embed_st.py"), "--latency",
                                model_id, str(gp), str(op)], check=True)
                times = json.loads(op.read_text(encoding="utf-8"))
        print(f"{name:<36} ms per tool output: median {q(times, .5):.0f}, "
              f"95th percentile {q(times, .95):.0f}, max {max(times):.0f} "
              f"(segments per output: median {statistics.median(len(g) for g in groups):.0f})")


def main(argv=None):
    p = argparse.ArgumentParser(description="Phrase-similarity candidates (step 2).")
    p.add_argument("--split", choices=["selection", "main"])
    p.add_argument("--robustness", action="store_true")
    p.add_argument("--latency", action="store_true")
    p.add_argument("--candidates", nargs="+", choices=CANDIDATES, default=list(CANDIDATES))
    args = p.parse_args(argv)
    for name in args.candidates:  # no phrase list may contain test phrasings
        hits = phrase_list.check_templates(CANDIDATES[name][0])
        if hits and name != "reference":
            raise SystemExit(f"{name}: phrases resemble test templates: {hits}")
    rows = eo.load_rows()
    if args.split == "selection":
        selection(args.candidates, rows)
    elif args.split == "main":
        main_split(args.candidates, rows)
    if args.robustness:
        robustness(args.candidates, rows)
    if args.latency:
        latency(args.candidates, rows)


if __name__ == "__main__":
    main()
