import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR      = Path(__file__).resolve().parent.parent.parent  # tools -> backend -> cineagent
ARTIFACTS_DIR = BASE_DIR / 'model_artifacts'
OUT_PATH      = ARTIFACTS_DIR / 'embed_matrix.npy'


def _safe_list(val, max_items=None):
    """
    Safely convert any column value to a flat list of strings.
    Handles: Python list, numpy array, comma-string, NaN, None.
    """
    if val is None:
        return []
    # numpy array (parquet stores list-columns this way)
    if isinstance(val, np.ndarray):
        items = val.tolist()
    elif isinstance(val, list):
        items = val
    else:
        s = str(val).strip()
        if not s or s.lower() in ('nan', 'none', '[]', ''):
            return []
        # Try JSON / Python literal first
        try:
            import ast
            parsed = ast.literal_eval(s)
            items = list(parsed) if isinstance(parsed, (list, tuple)) else [s]
        except Exception:
            # Fall back to comma-split
            items = [x.strip() for x in s.split(',') if x.strip()]
    items = [str(x).strip() for x in items if x and str(x).strip()
             and str(x).strip().lower() not in ('nan', 'none')]
    if max_items:
        items = items[:max_items]
    return items


def build_embed_text(row):
    """
    Build rich semantic text for embedding.

    Field weights (by repetition + position):
        genres   x3  — strongest vibe signal, short so needs boosting
        keywords x2  — thematic tags
        tagline  x2  — compact emotional hook ("In space no one can hear you scream")
        director x1  — stylistic fingerprint (Nolan films feel like Nolan films)
        cast     x1  — lead actor energy, top 3 only
        overview x1  — narrative description, truncated
        title    x1  — anchors the vector

    Why this beats title+overview only:
        - "Amelie" overview never uses the word "whimsical" but keywords do
        - Director "Jean-Pierre Jeunet" clusters with his other surreal films
        - Genres repeated 3x stops 300-char overview from drowning them out
    """
    title    = str(row.get('title',    '') or '').strip()
    overview = str(row.get('overview', '') or '')[:250].strip()
    tagline  = str(row.get('tagline',  '') or '').strip()
    director = str(row.get('director', '') or '').strip()

    # Genres: try genre_list first, fall back to genres string
    genre_items = _safe_list(row.get('genre_list')) or _safe_list(row.get('genres'))
    genres = ' '.join(genre_items).strip()

    # Keywords: try keyword_list first
    kw_items = _safe_list(row.get('keyword_list'), 20) or _safe_list(row.get('keywords'), 20)
    keywords = ' '.join(kw_items)[:200].strip()

    # Cast: top 3 names only — more creates noise
    cast_items = _safe_list(row.get('cast'), 3)
    cast = ' '.join(cast_items).strip()

    parts = []
    if title:
        parts.append(title)
    if genres:
        parts.extend([genres, genres, genres])       # x3 boost
    if keywords:
        parts.extend([keywords, keywords])            # x2 boost
    if tagline and tagline.lower() not in ('nan', 'none', ''):
        parts.extend([tagline, tagline])              # x2 boost
    if director and director.lower() not in ('nan', 'none', ''):
        parts.append(director)
    if cast:
        parts.append(cast)
    if overview:
        parts.append(overview)

    return ' | '.join(p for p in parts if p)


def main():
    print('[embed] Loading dataset...')
    parquet_path = ARTIFACTS_DIR / 'movies_enriched.parquet'
    if not parquet_path.exists():
        print('[embed] ERROR: movies_enriched.parquet not found at {}'.format(parquet_path))
        return

    df = pd.read_parquet(str(parquet_path))
    print('[embed] {:,} movies loaded'.format(len(df)))
    print('[embed] Columns: {}'.format(list(df.columns[:15])))

    print('[embed] Building embedding texts...')
    texts = df.apply(build_embed_text, axis=1).tolist()

    # Sanity-check a few samples
    for i in [0, 100, 1000]:
        if i < len(texts):
            title = df.iloc[i].get('title', '?')
            print('[embed] Sample [{}] "{}": {}...'.format(i, title, texts[i][:120]))

    print('[embed] Loading model: all-MiniLM-L6-v2...')
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print('[embed] ERROR: sentence-transformers not installed.')
        print('        Run: pip install sentence-transformers')
        return

    model = SentenceTransformer('all-MiniLM-L6-v2')
    print('[embed] Model loaded. Embedding dimension: 384')

    print('[embed] Generating embeddings — 5-15 minutes on CPU...')
    print('[embed] Using batch_size=64 (safe for 8 GB RAM). Increase to 128 if you have 16 GB+.')
    try:
        embeddings = model.encode(
            texts,
            batch_size           = 64,
            show_progress_bar    = True,
            normalize_embeddings = True,   # L2-normalize: dot product == cosine similarity
            convert_to_numpy     = True,
        )
    except MemoryError:
        print('[embed] MemoryError at batch_size=64 — retrying with batch_size=32...')
        embeddings = model.encode(
            texts,
            batch_size           = 32,
            show_progress_bar    = True,
            normalize_embeddings = True,
            convert_to_numpy     = True,
        )
    except Exception as e:
        print('[embed] ERROR during encoding: {}'.format(e))
        import traceback; traceback.print_exc()
        return
    embeddings = embeddings.astype(np.float32)

    print('[embed] Shape: {} (expected: {} x 384)'.format(embeddings.shape, len(df)))

    # Verify normalization (all norms should be ~1.0)
    norms = np.linalg.norm(embeddings[:100], axis=1)
    print('[embed] Norm check (first 100, should be ~1.0): min={:.4f} max={:.4f}'.format(
        norms.min(), norms.max()))

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(str(OUT_PATH), embeddings)
    size_mb = OUT_PATH.stat().st_size / 1024 / 1024
    print('[embed] Saved to {} ({:.1f} MB)'.format(OUT_PATH, size_mb))
    print('')
    print('[embed] Next step: python build_faiss_index.py')
    print('[embed] Then restart uvicorn.')


if __name__ == '__main__':
    main()