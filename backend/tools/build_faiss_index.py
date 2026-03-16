"""
build_faiss_index.py
Run AFTER generate_embeddings.py.
Loads embed_matrix.npy and builds a FAISS IndexFlatIP for fast cosine search.

Usage:
    cd C:/Users/91981/cineagent
    python build_faiss_index.py

Requires:
    pip install faiss-cpu

Output:
    model_artifacts/faiss_index.index  (~140 MB)
"""

import numpy as np
from pathlib import Path

BASE_DIR      = Path(__file__).resolve().parent.parent.parent
ARTIFACTS_DIR = BASE_DIR / 'model_artifacts'
EMBED_PATH    = ARTIFACTS_DIR / 'embed_matrix.npy'
INDEX_PATH    = ARTIFACTS_DIR / 'faiss_index.index'


def main():
    # ── Load embeddings ────────────────────────────────────────────────────────
    if not EMBED_PATH.exists():
        print('[faiss] ERROR: embed_matrix.npy not found at {}'.format(EMBED_PATH))
        print('        Run generate_embeddings.py first.')
        return

    print('[faiss] Loading embeddings...')
    embeddings = np.load(str(EMBED_PATH)).astype(np.float32)
    n, dim = embeddings.shape
    size_mb = embeddings.nbytes / 1024 / 1024
    print('[faiss] Shape: {} x {} ({:.1f} MB)'.format(n, dim, size_mb))

    # ── Import FAISS ───────────────────────────────────────────────────────────
    try:
        import faiss
    except ImportError:
        print('[faiss] ERROR: faiss-cpu not installed.')
        print('        Run: pip install faiss-cpu')
        return

    # ── Build index in chunks to avoid bad_alloc ───────────────────────────────
    # IndexFlatIP = exact inner product (= cosine sim on L2-normalised vectors)
    print('[faiss] Building IndexFlatIP (exact cosine search)...')
    index = faiss.IndexFlatIP(dim)

    CHUNK = 10000
    for start in range(0, n, CHUNK):
        end = min(start + CHUNK, n)
        index.add(embeddings[start:end])
        pct = int(end / n * 100)
        print('[faiss] Progress: {}/{} vectors  ({}%)'.format(end, n, pct))

    print('[faiss] Index contains {:,} vectors'.format(index.ntotal))

    # ── Verify with a quick self-search ───────────────────────────────────────
    query = embeddings[0:1]
    scores, ids = index.search(query, 3)
    print('[faiss] Self-search sanity check:')
    print('        top-1 id={} score={:.4f} (should be id=0, score=1.0)'.format(
        ids[0][0], scores[0][0]))

    # ── Save ───────────────────────────────────────────────────────────────────
    print('[faiss] Saving index to {}...'.format(INDEX_PATH))
    faiss.write_index(index, str(INDEX_PATH))
    size_mb = INDEX_PATH.stat().st_size / 1024 / 1024
    print('[faiss] Saved ({:.1f} MB)'.format(size_mb))
    print('')
    print('[faiss] Done! Restart uvicorn to use FAISS search.')


if __name__ == '__main__':
    main()