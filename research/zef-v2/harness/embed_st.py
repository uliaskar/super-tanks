"""
embed_st.py
===========
Embeds texts with a sentence-transformers model. Runs only inside the separate
environment harness/.venv-embed (PyTorch, sentence-transformers), so the shared
Super Tanks environment needs neither; embeddings.py and
eval_phrase_similarity.py call it.

Usage:
  .venv-embed/<python> embed_st.py MODEL TEXTS.json OUT.npy
  .venv-embed/<python> embed_st.py --latency MODEL GROUPS.json OUT.json
    GROUPS.json is a list of text lists (one per tool output); OUT.json gets the
    milliseconds to embed each group with the model already loaded.
"""

import json
import sys
import time

import numpy as np
from sentence_transformers import SentenceTransformer


def main():
    latency = sys.argv[1] == "--latency"
    model_name, in_path, out_path = sys.argv[2:5] if latency else sys.argv[1:4]
    with open(in_path, encoding="utf-8") as f:
        data = json.load(f)
    model = SentenceTransformer(model_name, device="cuda")
    if latency:
        model.encode(["warm-up"], normalize_embeddings=True)
        times = []
        for texts in data:
            t = time.perf_counter()
            model.encode(texts or ["empty"], normalize_embeddings=True, show_progress_bar=False)
            times.append((time.perf_counter() - t) * 1000)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(times, f)
        return
    vectors = model.encode(data, batch_size=128, normalize_embeddings=True,
                           show_progress_bar=False, convert_to_numpy=True)
    np.save(out_path, vectors.astype(np.float32))


if __name__ == "__main__":
    main()
