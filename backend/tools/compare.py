# backend/tools/compare.py
# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 3: compare_movies
# ═══════════════════════════════════════════════════════════════════════════════

import re
import json
import pandas as pd
from groq import Groq
from sklearn.metrics.pairwise import cosine_similarity

from .shared import (
    df, tfidf_matrix, GROQ_API_KEY,
    persist_if_new, check_disambiguation, resolve,
)

# Shared Groq client — instantiated once per server session
_groq_client = None
def _get_groq():
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client


def compare_movies(title1: str, title2: str,
                   year1: int = None, year2: int = None):
    """
    Compare two movies with year disambiguation, TF-IDF + Groq narrative.
    Returns (dict, error)
    """
    _size_before = len(df)

    def get_suggestions(title):
        """Partial title search — returns top 4 matches for disambiguation UI."""
        clean = re.sub(r'[^a-z0-9 ]', '', title.lower().strip())
        matches = df[df['title_clean'].str.contains(clean, na=False)]
        if matches.empty:
            return []
        return (matches.sort_values('vote_count', ascending=False)
                       .head(4)[['title', 'year', 'genres', 'vote_average', 'vote_count']]
                       .to_dict('records'))

    def resolve_with_year(title, year):
        clean = re.sub(r'[^a-z0-9 ]', '', title.lower().strip())

        # If year provided — partial+year search first, bypasses disambig entirely
        if year:
            candidates = df[
                (df['title_clean'].str.contains(clean, na=False)) &
                (df['year'] >= year - 1) &
                (df['year'] <= year + 1)
            ]
            if not candidates.empty:
                best = candidates['vote_count'].idxmax() if 'vote_count' in candidates.columns else candidates.index[0]
                print(f"[compare] Resolved '{title}' ({year}) → {df.loc[best, 'title']} ({df.loc[best, 'year']})")
                return int(best)
            print(f"[compare] No match for '{title}' in year {year} — trying resolve()")

        # No year — check disambiguation
        _, disambig = check_disambiguation(title)
        if disambig is not None:
            return 'DISAMBIG'

        # Try exact resolve
        idx = resolve(title)
        if idx is not None:
            return idx

        # Last resort — partial match, pick highest voted
        partial = df[df['title_clean'].str.contains(clean, na=False)]
        if not partial.empty:
            best = partial['vote_count'].idxmax() if 'vote_count' in partial.columns else partial.index[0]
            # Only auto-resolve if there's a clear winner (much higher votes than runner-up)
            if len(partial) == 1:
                print(f"[compare] Partial match: {df.loc[best, 'title']}")
                return int(best)
            # Multiple matches — treat as disambig
            return 'DISAMBIG'

        return None

    idx1 = resolve_with_year(title1, year1)
    idx2 = resolve_with_year(title2, year2)

    errs = []
    # For DISAMBIG cases, return structured options so frontend can show year picker
    if idx1 == 'DISAMBIG':
        suggestions = get_suggestions(title1)
        if suggestions:
            persist_if_new(_size_before)
            return None, f'DISAMBIG1:{json.dumps(suggestions)}'
        errs.append(f"'{title1}' is ambiguous — add year1=")
    if idx2 == 'DISAMBIG':
        suggestions = get_suggestions(title2)
        if suggestions:
            persist_if_new(_size_before)
            return None, f'DISAMBIG2:{json.dumps(suggestions)}'
        errs.append(f"'{title2}' is ambiguous — add year2=")
    if idx1 is None:       errs.append(f"Cannot resolve '{title1}'")
    if idx2 is None:       errs.append(f"Cannot resolve '{title2}'")
    if errs:
        persist_if_new(_size_before)
        return None, ' | '.join(errs)

    # FIX: use .loc not .iloc
    if idx1 not in df.index or idx2 not in df.index:
        return None, "Index out of bounds after injection — try again."

    r1  = df.loc[idx1]
    r2  = df.loc[idx2]

    # FIX: get positional index for tfidf_matrix (which is position-based)
    pos1 = df.index.get_loc(idx1)
    pos2 = df.index.get_loc(idx2)
    sim  = cosine_similarity(tfidf_matrix[pos1], tfidf_matrix[pos2]).flatten()[0]

    def _to_set(val):
        import numpy as np
        if val is None: return set()
        if isinstance(val, np.ndarray): return set(val.tolist())
        if isinstance(val, (list, set)): return set(val)
        return set()
    g1 = _to_set(r1.get('genre_list'))
    g2 = _to_set(r2.get('genre_list'))
    k1 = _to_set(r1.get('keyword_list'))
    k2 = _to_set(r2.get('keyword_list'))

    try:
        y1 = int(r1['year']) if r1['year'] is not None and not pd.isna(r1['year']) else None
    except (TypeError, ValueError):
        y1 = None
    try:
        y2 = int(r2['year']) if r2['year'] is not None and not pd.isna(r2['year']) else None
    except (TypeError, ValueError):
        y2 = None

    # ── Groq narrative comparison ─────────────────────────────────────────────
    groq_analysis = None
    if GROQ_API_KEY and GROQ_API_KEY != 'YOUR_GROQ_KEY_HERE':
        prompt = (
            f"Compare these two movies for someone deciding what to watch:\n\n"
            f"Movie 1: {r1['title']} ({y1}) — {r1.get('genres','')} — {r1.get('overview','')[:150]}\n"
            f"Movie 2: {r2['title']} ({y2}) — {r2.get('genres','')} — {r2.get('overview','')[:150]}\n\n"
            f"Return ONLY a JSON object:\n"
            f"{{\n"
            f'  "similarity_summary": "one sentence on how similar they are",\n'
            f'  "shared_themes":      ["theme1", "theme2"],\n'
            f'  "tone_movie1":        "e.g. dark and intense",\n'
            f'  "tone_movie2":        "e.g. light and hopeful",\n'
            f'  "watch_movie1_if":    "type of viewer who would prefer movie 1",\n'
            f'  "watch_movie2_if":    "type of viewer who would prefer movie 2",\n'
            f'  "verdict":            "3-4 sentences. Compare them directly — what each does better, which has stronger writing/direction/performances, and which to watch first. Be clear and opinionated. No wishy-washy hedging. Give a real recommendation."\n'
            f"}}"
        )
        try:
            resp = _get_groq().chat.completions.create(
                model='llama-3.3-70b-versatile',
                messages=[{'role': 'user', 'content': prompt}],
                temperature=0.1,
                max_tokens=600,
            )
            raw = resp.choices[0].message.content.strip()
            raw = re.sub(r'^```json?\s*', '', raw, flags=re.MULTILINE)
            raw = re.sub(r'```\s*$',      '', raw, flags=re.MULTILINE)
            groq_analysis = json.loads(raw.strip())
        except Exception as e:
            print(f"[compare] Groq failed: {e}")

    # FIX: structure as movie1/movie2 dicts so frontend render_compare() works
    result = {
        'movie1': {
            'title':        r1['title'],
            'year':         y1,
            'vote_average': round(float(pd.to_numeric(r1.get('vote_average', 0), errors='coerce') or 0), 2),
            'vote_count':   float(pd.to_numeric(r1.get('vote_count', 0),   errors='coerce') or 0),
            'genres':       r1.get('genres', '') or '',
            'director':     r1.get('director', '') or '',
            'overview':     r1.get('overview', '') or '',
        },
        'movie2': {
            'title':        r2['title'],
            'year':         y2,
            'vote_average': round(float(pd.to_numeric(r2.get('vote_average', 0), errors='coerce') or 0), 2),
            'vote_count':   float(pd.to_numeric(r2.get('vote_count', 0),   errors='coerce') or 0),
            'genres':       r2.get('genres', '') or '',
            'director':     r2.get('director', '') or '',
            'overview':     r2.get('overview', '') or '',
        },
        'tfidf_similarity': round(float(sim), 4),
        'shared_genres':    list(g1 & g2),
        'unique_to_movie1': list(g1 - g2),
        'unique_to_movie2': list(g2 - g1),
        'shared_keywords':  list(k1 & k2)[:8],
    }

    if groq_analysis:
        result.update(groq_analysis)

    persist_if_new(_size_before)
    return result, None