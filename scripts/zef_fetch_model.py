"""
scripts/zef_fetch_model.py
===========================
Downloads the model files ZEF's phrase similarity needs into data/zef/:
the official ONNX export of paraphrase-multilingual-MiniLM-L12-v2 and its
tokenizer, at the revision and SHA-256 pinned in
core/security/zef_phrase_similarity.py. A file whose checksum does not match
is deleted, never kept. About 480 MB; run once.

Usage
-----
    python -m scripts.zef_fetch_model

Exit codes: 0 all files present and verified, 1 a download or check failed.
"""

from __future__ import annotations

import hashlib
import sys

import requests

from core.security.zef_combine import MODEL_DIR
from core.security.zef_phrase_similarity import MODEL_FILES, REPO, REVISION

URL = "https://huggingface.co/{repo}/resolve/{revision}/{path}"


def _sha256(path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(name: str, remote: str, sha256: str) -> bool:
    target = MODEL_DIR / name
    if target.exists() and _sha256(target) == sha256:
        print(f"{name}: present, checksum OK")
        return True
    tmp = target.with_suffix(target.suffix + ".part")
    url = URL.format(repo=REPO, revision=REVISION, path=remote)
    print(f"{name}: downloading {url}")
    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with tmp.open("wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
    except requests.RequestException as e:
        tmp.unlink(missing_ok=True)
        print(f"{name}: download failed: {e}")
        return False
    if _sha256(tmp) != sha256:
        tmp.unlink()
        print(f"{name}: checksum does not match the pinned one; file deleted")
        return False
    tmp.replace(target)
    print(f"{name}: saved, checksum OK")
    return True


def main() -> int:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ok = all([fetch(name, remote, sha) for name, (remote, sha) in MODEL_FILES.items()])
    print(f"model files in {MODEL_DIR}: {'ready' if ok else 'incomplete'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
