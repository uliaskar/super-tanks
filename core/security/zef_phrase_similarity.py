"""
core/security/zef_phrase_similarity.py
=======================================
ZEF phrase similarity: finds rewordings and translations of known injection
phrases by meaning.

Segments and phrases are embedded with paraphrase-multilingual-MiniLM-L12-v2
(the official ONNX export, pinned by revision and SHA-256) and compared by
cosine; a segment's score is its highest similarity to any phrase. Both are
lowercased. Embedding follows sentence-transformers: at most MAX_TOKENS
tokens, mean of the token vectors over the attention mask, length 1. Runs
on ONNX Runtime, on the GPU when onnxruntime-gpu and CUDA are available,
else on the CPU (same results). Measured in research/zef-v2
(eval_phrase_similarity.py).

Needs the optional extra `super-tanks[zef]` (or `[zef-gpu]`) and the model
files fetched by scripts/zef_fetch_model.py.
"""

import logging
from pathlib import Path

logger = logging.getLogger("zef.phrase_similarity")

REPO = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
# file in data/zef/ -> (path in the repository, SHA-256)
MODEL_FILES = {
    "model.onnx": ("onnx/model.onnx",
                   "10f7a088420252b26caf819236ca2c9d2987afd0fc06fec7553b542a5655a05a"),
    "tokenizer.json": ("tokenizer.json",
                       "2c3387be76557bd40970cec13153b3bbf80407865484b209e655e5e4729076b8"),
}
MAX_TOKENS = 128
DIM = 384
BATCH = 32
NO_SEGMENT = -1.0  # score when there is nothing to judge


class PhraseSimilarity:
    def __init__(self, model_dir: Path, phrases: list[str]):
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self._np = np
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                     if p in ort.get_available_providers()]
        self._session = ort.InferenceSession(str(model_dir / "model.onnx"), providers=providers)
        self._inputs = {i.name for i in self._session.get_inputs()}
        self._tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self._tokenizer.enable_truncation(MAX_TOKENS)
        self._tokenizer.enable_padding()
        self._phrases = self.embed([p.lower() for p in phrases])
        logger.info("[ZEF] phrase similarity on %s", self._session.get_providers()[0])

    def embed(self, texts: list[str]):
        """Unit vectors, one per text; batches sorted by length as sentence-transformers does."""
        np = self._np
        out = np.zeros((len(texts), DIM), np.float32)
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        for i in range(0, len(order), BATCH):
            idx = order[i:i + BATCH]
            enc = self._tokenizer.encode_batch([texts[j] for j in idx])
            mask = np.array([e.attention_mask for e in enc], np.int64)
            feed = {"input_ids": np.array([e.ids for e in enc], np.int64), "attention_mask": mask}
            if "token_type_ids" in self._inputs:
                feed["token_type_ids"] = np.array([e.type_ids for e in enc], np.int64)
            tokens = self._session.run(None, feed)[0]
            m = mask[..., None].astype(np.float32)
            v = (tokens * m).sum(1) / np.maximum(m.sum(1), 1e-9)
            out[idx] = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
        return out

    def each(self, segments: list[str]):
        """Per segment, its highest similarity to a phrase."""
        return (self.embed([s.lower() for s in segments]) @ self._phrases.T).max(axis=1)

    def best(self, segments: list[str]) -> tuple[float, str | None]:
        """Highest similarity and its segment (NO_SEGMENT, None without segments)."""
        if not segments:
            return NO_SEGMENT, None
        per_segment = self.each(segments)
        i = int(per_segment.argmax())
        return float(per_segment[i]), segments[i]
