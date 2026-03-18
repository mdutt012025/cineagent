# backend/tools/shared.py
import numpy as np
import pandas as pd
import scipy.sparse
import joblib
import re
import unicodedata
import json
import ast
import os
import requests
from datetime import datetime
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR      = Path("/app")
ARTIFACTS_DIR = Path("/app/model_artifacts")

GROQ_API_KEY = os.environ.get('GROQ_API_KEY', 'YOUR_GROQ_KEY_HERE')
TMDB_API_KEY = os.environ.get('TMDB_API_KEY', 'YOUR_TMDB_KEY_HERE')

CURRENT_YEAR = datetime.now().year

# ── Shared Groq client — instantiated once per server session ─────────────────
from groq import Groq as _Groq
_groq_client = None
def _get_groq_client():
    global _groq_client
    if _groq_client is None:
        _groq_client = _Groq(api_key=GROQ_API_KEY)
    return _groq_client

print("[shared] Loading artifacts...")
df = pd.read_parquet(ARTIFACTS_DIR / 'movies_enriched.parquet')

def parse_genre_list(row):
    val = row['genre_list']
    if isinstance(val, list) and val:
        return val
    if isinstance(val, str) and val.strip().startswith('['):
        try:
            parsed = ast.literal_eval(val)
            if parsed:
                return parsed
        except:
            pass
    genres_str = row.get('genres') or ''
    if isinstance(genres_str, str) and genres_str.strip():
        return [g.strip() for g in genres_str.split(',') if g.strip()]
    return []

df['genre_list'] = df.apply(parse_genre_list, axis=1)
tfidf        = joblib.load(ARTIFACTS_DIR / 'tfidf_vectorizer.joblib')
tfidf_matrix = joblib.load(ARTIFACTS_DIR / 'tfidf_matrix.joblib')
title_to_idx = joblib.load(ARTIFACTS_DIR / 'title_to_idx.joblib')

# ── Semantic search: FAISS index (preferred) or raw embeddings (fallback) ─────
embed_matrix = None
faiss_index  = None
_EMBED_PATH  = ARTIFACTS_DIR / 'embed_matrix.npy'
_FAISS_PATH  = ARTIFACTS_DIR / 'faiss_index.index'

if _FAISS_PATH.exists():
    try:
        import faiss as _faiss
        faiss_index = _faiss.read_index(str(_FAISS_PATH))
        if faiss_index.ntotal != len(df):
            print('[shared] FAISS size mismatch ({} vs {}) — rebuild recommended'.format(
                faiss_index.ntotal, len(df)))
            faiss_index = None
        else:
            print('[shared] FAISS index loaded: {:,} vectors dim={}'.format(
                faiss_index.ntotal, faiss_index.d))
    except ImportError:
        print('[shared] faiss-cpu not installed — pip install faiss-cpu')
        faiss_index = None
    except Exception as e:
        print('[shared] FAISS load failed: {}'.format(e))
        faiss_index = None

if faiss_index is None and _EMBED_PATH.exists():
    try:
        embed_matrix = np.load(str(_EMBED_PATH), mmap_mode='r')
        if embed_matrix.shape[0] != len(df):
            _n           = min(embed_matrix.shape[0], len(df))
            embed_matrix = embed_matrix[:_n]
        print('[shared] Embeddings loaded (numpy): {} x {}'.format(*embed_matrix.shape))
        print('[shared] Run python build_faiss_index.py for faster search')
    except Exception as e:
        print('[shared] Embed matrix load failed: {}'.format(e))
        embed_matrix = None

if faiss_index is None and embed_matrix is None:
    print('[shared] No semantic index — using TF-IDF only')
    print('[shared] Run: python generate_embeddings.py  then  python build_faiss_index.py')

# ── Normalized title index — handles accented titles ─────────────────────────
import unicodedata as _ud
_normalized_to_idx = {}
if 'title' in df.columns:
    for _i, _t in enumerate(df['title'].fillna('')):
        try:
            _nfd  = _ud.normalize('NFD', str(_t))
            _norm = re.sub(r'[^a-z0-9 ]', '', _nfd.encode('ascii','ignore').decode('ascii').lower().strip())
            _raw  = re.sub(r'[^a-z0-9 ]', '', str(_t).lower().strip())
            if _norm != _raw and _norm not in title_to_idx and _norm not in _normalized_to_idx:
                _normalized_to_idx[_norm] = _i
        except Exception:
            pass
del _ud

print("[shared] Loaded — {:,} movies, matrix {}".format(len(df), tfidf_matrix.shape))

# ── Startup sync check ────────────────────────────────────────────────────────
if len(df) != tfidf_matrix.shape[0]:
    print("[shared] WARNING: df/matrix out of sync ({} vs {}) — resyncing".format(
        len(df), tfidf_matrix.shape[0]))
    n            = min(len(df), tfidf_matrix.shape[0])
    df           = df.iloc[:n].reset_index(drop=True)
    tfidf_matrix = tfidf_matrix[:n]
    title_to_idx = pd.Series(df.index, index=df['title_clean']).to_dict()
    print("[shared] Resynced to {:,} rows".format(n))

_SCRAPE_CACHE          = {}
_INJECTED_THIS_SESSION = set()
_VIBE_CACHE            = {}  # max 100 entries — evict oldest on overflow
_VIBE_CACHE_MAX        = 100

LAST_SESSION = {
    'tool': None, 'query': None, 'shown': [],
    'era': None, 'min_votes': 500, 'top_n': 5,
}

ERA_MAP = {
    'classic': (None, 1989), '90s': (1990, 1999),
    '2000s': (2000, 2009),   '2010s': (2010, 2017),
    'modern': (2018, 2021),  'new': (2022, None),
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def _safe_save():
    save_df = df.copy()
    for col in save_df.columns:
        if col in ('genre_list', 'keyword_list'):
            continue
        if save_df[col].dtype == object:
            save_df[col] = save_df[col].apply(
                lambda x: str(x) if not isinstance(x, (str, type(None), float, int)) else x
            )
    save_df.to_parquet(ARTIFACTS_DIR / 'movies_enriched.parquet', index=False)
    joblib.dump(tfidf_matrix, ARTIFACTS_DIR / 'tfidf_matrix.joblib')
    joblib.dump(dict(title_to_idx), ARTIFACTS_DIR / 'title_to_idx.joblib')


def safe_row(idx):
    """
    Safely fetch a df row by index label.
    After concat+reset_index, labels are always 0..N-1.
    If idx is out of range (stale label), clamp to valid range.
    """
    if idx in df.index:
        return df.loc[idx].to_dict()
    pos = min(int(idx), len(df) - 1)
    print("[safe_row] Stale idx {} — using iloc[{}]".format(idx, pos))
    return df.iloc[pos].to_dict()


def persist_if_new(size_before):
    if len(df) > size_before:
        _safe_save()
        print("[persist] {} new movie(s) saved — total {:,}".format(len(df) - size_before, len(df)))


def looks_like_gibberish(title):
    t = title.lower().strip()
    if len(t) < 2: return True
    if re.search(r'[a-z]{4,}\d+', t) or re.search(r'\d+[a-z]{4,}', t): return True
    letters = re.sub(r'[^a-z]', '', t)
    if len(letters) > 6 and sum(1 for c in letters if c in 'aeiou') / len(letters) < 0.12:
        return True
    if len(t) > 8 and ' ' not in t and sum(1 for c in t if c in 'aeiou') / len(t) < 0.15:
        return True
    return False


def is_compound_mood(mood):
    words = mood.lower().strip().split()
    if len(words) == 1: return False
    genre_words = {'com','fi','comedy','romance','action','horror','thriller','drama','sci','rom'}
    return len(set(words) & genre_words) >= 2 or len(words) >= 3


def _normalize_title(title):
    """Strip accents then non-alphanumeric. 'Amélie' -> 'amelie'."""
    try:
        nfd = unicodedata.normalize('NFD', str(title))
        asc = nfd.encode('ascii', 'ignore').decode('ascii')
    except Exception:
        asc = str(title)
    return re.sub(r'[^a-z0-9 ]', '', asc.lower().strip())


def check_disambiguation(title):
    clean   = _normalize_title(title)
    matches = df[df['title_clean'] == clean].copy()
    if matches.empty:
        clean_raw = re.sub(r'[^a-z0-9 ]', '', title.lower().strip())
        if clean_raw != clean:
            matches = df[df['title_clean'] == clean_raw].copy()
    if len(matches) > 1:
        sort_col = 'vote_count' if 'vote_count' in matches.columns else 'vote_average'
        matches  = matches.sort_values(sort_col, ascending=False)
        return None, matches[['title','year','genres','vote_average','vote_count']].reset_index(drop=True)
    return None, None


def franchise_key(t):
    t = str(t).lower()
    t = re.sub(r'\b(vol\.?|volume|part|chapter|episode|season)\s*[\d\w]+.*$', '', t)
    t = re.sub(r'\s+\d+\s*$', '', t)
    t = re.sub(r':\s*.+$', '', t)
    t = re.sub(r'\s+(ii|iii|iv|v|vi|vii)\s*$', '', t)
    return t.strip()


def get_era_range(era):
    if not era: return 1970, None
    ec = era.lower().strip()
    if ec in ERA_MAP: return ERA_MAP[ec]
    if ec.isdigit():
        yr = int(ec)
        return yr - 5, yr + 5
    return 1970, None


# ── TMDB ───────────────────────────────────────────────────────────────────────

TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_IMG  = "https://image.tmdb.org/t/p/w500"


def tmdb_search(title, year=None):
    if not TMDB_API_KEY or TMDB_API_KEY == 'YOUR_TMDB_KEY_HERE':
        return {'error': 'No TMDB key'}
    try:
        params = {'api_key': TMDB_API_KEY, 'query': title,
                  'include_adult': False, 'language': 'en-US'}
        if year:
            params['year'] = year
        results = requests.get(
            "{}/search/movie".format(TMDB_BASE), params=params, timeout=8
        ).json().get('results', [])
        if not results:
            return {'error': 'No TMDB results for "{}"'.format(title)}

        best = results[0]
        for r in results:
            if r.get('title', '').lower() == title.lower():
                best = r
                break

        detail = requests.get(
            "{}/movie/{}".format(TMDB_BASE, best['id']),
            params={'api_key': TMDB_API_KEY, 'append_to_response': 'credits,keywords'},
            timeout=8,
        ).json()

        crew    = detail.get('credits', {}).get('crew', [])
        cast    = detail.get('credits', {}).get('cast', [])
        release = detail.get('release_date', '')

        return {
            'tmdb_id':           str(best['id']),
            'title':             detail.get('title', title),
            'year':              int(release[:4]) if release and len(release) >= 4 else None,
            'genres':            [g['name'] for g in detail.get('genres', [])],
            'keywords':          [k['name'] for k in detail.get('keywords', {}).get('keywords', [])][:20],
            'director':          next((c['name'] for c in crew if c.get('job') == 'Director'), ''),
            'cast':              [c['name'] for c in cast[:5]],
            'overview':          detail.get('overview', ''),
            'tagline':           detail.get('tagline', ''),
            'vote_average':      round(detail.get('vote_average', 0), 2),
            'vote_count':        detail.get('vote_count', 0),
            'popularity':        round(detail.get('popularity', 0), 2),
            'poster_url':        "{}{}".format(TMDB_IMG, detail['poster_path']) if detail.get('poster_path') else '',
            'imdb_id':           str(detail.get('imdb_id', '')),
            'original_language': detail.get('original_language', ''),
        }
    except Exception as e:
        return {'error': 'TMDB failed: {}'.format(e)}


def tmdb_enrich_rating(title, year=None):
    """Fetch just vote_average + vote_count + poster from TMDB for injected films."""
    result = tmdb_search(title, year)
    if 'error' in result:
        return 0.0, 0, ''
    return (
        float(result.get('vote_average') or 0),
        int(result.get('vote_count') or 0),
        result.get('poster_url', ''),
    )


# ── Groq Enrich ────────────────────────────────────────────────────────────────

def groq_enrich(title, tmdb_data):
    if not GROQ_API_KEY or GROQ_API_KEY == 'YOUR_GROQ_KEY_HERE':
        return tmdb_data if tmdb_data and 'error' not in tmdb_data else None
    prompt = (
        'You are a movie metadata expert. Given partial metadata for "{}", '
        'return a complete normalised JSON object.\n\n'
        'Partial metadata:\n{}\n\n'
        'IMPORTANT: If this does not appear to be a real movie title, '
        'return exactly: {{"error": "not_a_movie"}}\n\n'
        'Otherwise return ONLY valid JSON with exactly these keys:\n'
        '{{"title":"","year":2023,"genres":[],"keywords":[],"director":"",'
        '"cast":[],"overview":"","vote_average":7.5,"popularity":30.0,"tagline":""}}'
    ).format(title, json.dumps(tmdb_data, indent=2))
    try:
        resp = _get_groq_client().chat.completions.create(
            model='llama-3.3-70b-versatile',
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0.1, max_tokens=600,
        )
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r'^```json?\s*', '', raw, flags=re.MULTILINE)
        raw = re.sub(r'```\s*$', '', raw, flags=re.MULTILINE)
        result = json.loads(raw.strip())
        if result.get('error') == 'not_a_movie':
            return None
        for field in ('poster_url', 'tmdb_id', 'imdb_id', 'vote_count', 'original_language'):
            if field in tmdb_data:
                result[field] = tmdb_data[field]
        return result
    except Exception as e:
        print('[Groq enrich] Error: {}'.format(e))
        return tmdb_data if tmdb_data and 'error' not in tmdb_data else None


# ── Inject ─────────────────────────────────────────────────────────────────────

def inject_movie(meta, save=True):
    """Inject a movie into df + tfidf_matrix + title_to_idx."""
    global df, tfidf_matrix, title_to_idx

    title_raw   = meta.get('title', 'Unknown')
    title_clean = re.sub(r'[^a-z0-9 ]', '', title_raw.lower().strip())

    if title_clean in title_to_idx:
        stored_idx = title_to_idx[title_clean]
        if stored_idx < len(df):
            stored_clean = re.sub(r'[^a-z0-9 ]', '', str(df.iloc[stored_idx].get('title', '')).lower().strip())
            if stored_clean == title_clean:
                print("[inject] '{}' already exists — skipping".format(title_raw))
                return stored_idx
        matches = df[df['title_clean'] == title_clean]
        if not matches.empty:
            correct_idx = int(matches.index[-1])
            title_to_idx[title_clean] = correct_idx
            return correct_idx

    genres   = meta.get('genres', [])
    keywords = meta.get('keywords', [])
    director = meta.get('director', '')
    if isinstance(director, list):
        director = director[0] if director else ''

    def tok(s): return re.sub(r'\s+', '_', str(s).strip())
    genre_str   = ' '.join(tok(g) for g in genres)
    keyword_str = ' '.join(tok(k) for k in keywords)
    overview    = meta.get('overview', '')
    tagline     = meta.get('tagline', '')
    parts       = [genre_str]*3 + [keyword_str]*2 + [overview]*4 + ([tagline]*2 if tagline and tagline != 'nan' else [])
    content_soup = ' '.join(p for p in parts if p)

    year       = float(meta['year']) if meta.get('year') else np.nan
    vote_avg   = float(meta.get('vote_average') or 0)
    vote_count = float(meta.get('vote_count') or 0)
    popularity = float(meta.get('popularity') or 0)
    poster_url = str(meta.get('poster_url', '') or '')

    # If Groq-only meta has no real rating, fetch from TMDB
    if vote_avg == 0 or vote_count == 0:
        print("[inject] '{}' has no rating — fetching from TMDB...".format(title_raw))
        real_avg, real_count, real_poster = tmdb_enrich_rating(
            title_raw, int(year) if not np.isnan(year) else None
        )
        if real_avg > 0:
            vote_avg   = real_avg
            vote_count = float(real_count)
            print("[inject] Got real rating: ★{:.1f} ({:,} votes)".format(vote_avg, int(vote_count)))
        if real_poster and not poster_url:
            poster_url = real_poster

    new_row = {
        'id':                  -1,
        'title':               title_raw,
        'title_clean':         title_clean,
        'original_language':   str(meta.get('original_language', '')),
        'year':                year,
        'genres':              ', '.join(genres) if isinstance(genres, list) else genres,
        'genre_list':          genres,
        'genre_str':           genre_str,
        'keyword_list':        keywords,
        'keyword_str':         keyword_str,
        'keywords':            ', '.join(keywords) if isinstance(keywords, list) else keywords,
        'director':            director,
        'director_str':        tok(director),
        'overview':            overview,
        'tagline':             tagline,
        'vote_average':        vote_avg,
        'vote_count':          vote_count,
        'popularity':          popularity,
        'weighted_score':      vote_avg,
        'weighted_score_norm': vote_avg / 10,
        'popularity_norm':     min(popularity / 100, 1.0),
        'hidden_gem_score':    (vote_avg / 10) - 0.5 * min(popularity / 100, 1.0),
        'content_soup':        content_soup,
        'release_date':        str(meta.get('year', '')),
        'poster_url':          poster_url,
        'tmdb_id':             str(meta.get('tmdb_id', '') or ''),
        'imdb_id':             str(meta.get('imdb_id', '') or ''),
        'groq_injected':       True,
    }

    df           = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.reset_index(drop=True, inplace=True)
    new_vec      = tfidf.transform([content_soup])
    tfidf_matrix = scipy.sparse.vstack([tfidf_matrix, new_vec])
    new_idx      = len(df) - 1

    title_to_idx[title_clean] = new_idx
    _INJECTED_THIS_SESSION.add(title_clean)

    print("[inject] '{}' ({}) ★{:.1f} added — total {:,}".format(
        title_raw, int(year) if not np.isnan(year) else '?', vote_avg, len(df)))

    if save:
        _safe_save()

    return new_idx


# ── Resolve ────────────────────────────────────────────────────────────────────

def resolve(title, save_on_inject=True):
    clean    = _normalize_title(title)
    clean_ny = re.sub(r'\s*\d{4}\s*$', '', clean).strip()
    clean_raw = re.sub(r'[^a-z0-9 ]', '', title.lower().strip())

    # 1. Exact DB match
    for key in (clean, clean_ny, clean_raw):
        if not key:
            continue
        if key not in title_to_idx and key in _normalized_to_idx:
            idx_val = _normalized_to_idx[key]
            if idx_val < len(df):
                return int(idx_val)
        matches = df[df['title_clean'] == key]
        if matches.empty and key in _normalized_to_idx:
            stored_idx = _normalized_to_idx[key]
            if stored_idx < len(df):
                return int(stored_idx)
        if not matches.empty:
            if len(matches) > 1 and 'vote_count' in matches.columns:
                best = matches['vote_count'].idxmax()
                title_to_idx[key] = int(best)
                return int(best)
            title_to_idx[key] = int(matches.index[0])
            return int(matches.index[0])

    # 2. Skip gibberish
    if looks_like_gibberish(title):
        return None

    # 3. TMDB fetch
    if title not in _SCRAPE_CACHE:
        print("[resolve] '{}' not in DB — fetching TMDB...".format(title))
        tmdb_data = tmdb_search(title)
        _SCRAPE_CACHE[title] = tmdb_data
    else:
        tmdb_data = _SCRAPE_CACHE[title]

    if 'error' not in tmdb_data and tmdb_data.get('title'):
        tmdb_title = tmdb_data['title']
        tmdb_clean = re.sub(r'[^a-z0-9 ]', '', tmdb_title.lower().strip())
        import unicodedata as _ud2
        _nfd2      = _ud2.normalize('NFD', str(tmdb_title))
        tmdb_norm  = re.sub(r'[^a-z0-9 ]', '', _nfd2.encode('ascii','ignore').decode('ascii').lower().strip())
        for _chk in (tmdb_clean, tmdb_norm):
            if _chk in title_to_idx:
                return int(title_to_idx[_chk])
            if _chk in _normalized_to_idx:
                return int(_normalized_to_idx[_chk])
        # Only enrich via Groq if TMDB data is missing key fields
        _needs_enrich = not (tmdb_data.get('genres') and tmdb_data.get('overview') and tmdb_data.get('director'))
        if _needs_enrich:
            meta = groq_enrich(title, tmdb_data)
        else:
            meta = tmdb_data
        if meta:
            meta.setdefault('title', tmdb_title)
            return inject_movie(meta, save=save_on_inject)

    # 4. Fuzzy match — last resort
    from difflib import get_close_matches
    close = get_close_matches(clean_ny, list(title_to_idx.keys()), n=1, cutoff=0.85)
    if close:
        if set(clean_ny.split()) & set(close[0].split()):
            idx = int(title_to_idx[close[0]])
            print("[resolve] fuzzy: '{}' → '{}'".format(clean_ny, df.loc[idx, 'title']))
            return idx

    return None


def groq_parse_refinement(refine_text, last_session):
    prompt = (
        "A user is refining movie recommendations.\n"
        "Context: tool={}, query={}, shown={}\n"
        "User said: '{}'\n\n"
        "Return ONLY JSON:\n"
        '{{"action":"more"|"constrain"|"new_mood","new_mood":null,"genres_exclude":[],"genres_require":[],"keywords_avoid":[]}}'
    ).format(
        last_session.get('tool'), last_session.get('query'),
        last_session.get('shown'), refine_text
    )
    try:
        resp = _get_groq_client().chat.completions.create(
            model='llama-3.1-8b-instant',
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0.1, max_tokens=200,
        )
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r'^```json?\s*', '', raw, flags=re.MULTILINE)
        raw = re.sub(r'```\s*$', '', raw, flags=re.MULTILINE)
        return json.loads(raw.strip())
    except Exception as e:
        print("[refine] Error: {}".format(e))
        return {'action': 'more', 'new_mood': None, 'genres_exclude': [],
                'genres_require': [], 'keywords_avoid': []}