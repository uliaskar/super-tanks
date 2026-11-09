"""
embeddings.py
=============
Embedding models compared in step 2 (../README.md), behind one function, with
a cache per model in data/emb/ so no text is embedded twice.

  * Ollama models are called over Ollama's HTTP API (OLLAMA_LOCAL_HOST /
    OLLAMA_LOCAL_PORT, default localhost:11434; on Windows use 127.0.0.1).
  * sentence-transformers models run in harness/.venv-embed via embed_st.py.

Every vector is returned with length 1, so a dot product is the cosine.
The prefix is what each model card asks for when comparing texts of the same
kind; it is part of the cached text.
"""

import hashlib
import json
import os
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CACHE_DIR = HERE / "data" / "emb"
EMBED_ENV = HERE / ".venv-embed" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
BATCH = 64

# name -> (backend, model id, prefix)
MODELS = {
    "nomic-embed-text": ("ollama", "nomic-embed-text", "clustering: "),
    "bge-m3": ("ollama", "bge-m3", ""),
    "snowflake-arctic-embed2": ("ollama", "snowflake-arctic-embed2", ""),
    "granite-embedding-278m": ("ollama", "granite-embedding:278m", ""),
    "paraphrase-multilingual-minilm": (
        "st", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", ""),
    "multilingual-e5-small": ("st", "intfloat/multilingual-e5-small", "query: "),
    "labse": ("st", "sentence-transformers/LaBSE", ""),
}


def _key(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _unit(m):
    m = np.asarray(m, dtype=np.float32)
    return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12)


def _ollama(model_id, texts, attempts=5):
    host = os.getenv("OLLAMA_LOCAL_HOST", "localhost")
    port = os.getenv("OLLAMA_LOCAL_PORT", "11434")
    body = json.dumps({"model": model_id, "input": texts, "keep_alive": "30m"}).encode("utf-8")
    req = urllib.request.Request(f"http://{host}:{port}/api/embed", data=body,
                                 headers={"Content-Type": "application/json"})
    # Ollama rejects a batch (HTTP 400) while it unloads or reloads a model,
    # which happens when several models compete for GPU memory: wait and retry.
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                vectors = json.loads(resp.read().decode("utf-8"))["embeddings"]
            break
        except urllib.error.HTTPError:
            if attempt == attempts - 1:
                raise
            time.sleep(5 * (attempt + 1))
    if len(vectors) != len(texts):
        raise RuntimeError(f"{model_id}: {len(vectors)} vectors for {len(texts)} texts")
    return vectors


def _sentence_transformers(model_id, texts):
    with tempfile.TemporaryDirectory() as tmp:
        texts_path, out_path = Path(tmp) / "texts.json", Path(tmp) / "out.npy"
        texts_path.write_text(json.dumps(texts, ensure_ascii=False), encoding="utf-8")
        subprocess.run([str(EMBED_ENV), str(HERE / "embed_st.py"), model_id,
                        str(texts_path), str(out_path)], check=True)
        return np.load(out_path)


class Embedder:
    """Cached embeddings of one model."""

    def __init__(self, name):
        self.name = name
        self.backend, self.model_id, self.prefix = MODELS[name]
        self.path = CACHE_DIR / f"{name}.npz"
        self.cache = {}
        if self.path.exists():
            data = np.load(self.path)
            self.cache = dict(zip(data["keys"].tolist(), data["vectors"]))

    def save(self):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        keys = list(self.cache)
        np.savez(self.path, keys=np.array(keys), vectors=np.stack([self.cache[k] for k in keys]))

    def __call__(self, texts, save=True):
        """Unit vectors for texts (missing ones are embedded and cached)."""
        full = [self.prefix + t for t in texts]
        todo = sorted({t for t in full if _key(t) not in self.cache})
        if todo:
            if self.backend == "ollama":
                for i in range(0, len(todo), BATCH):
                    batch = todo[i:i + BATCH]
                    for t, v in zip(batch, _unit(_ollama(self.model_id, batch))):
                        self.cache[_key(t)] = v
                    if save and (i // BATCH) % 200 == 199:  # keep progress on long runs
                        self.save()
            else:
                for t, v in zip(todo, _unit(_sentence_transformers(self.model_id, todo))):
                    self.cache[_key(t)] = v
            if save:
                self.save()
        if not full:
            return np.zeros((0, 1), dtype=np.float32)
        return np.stack([self.cache[_key(t)] for t in full])
