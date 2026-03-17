# backend/tools/mood.py
# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 2: mood_based_recommend
# ═══════════════════════════════════════════════════════════════════════════════

import re
import json
import time as _time
import pandas as pd
import numpy as np
from groq import Groq

from .shared import (
    df, GROQ_API_KEY, LAST_SESSION,
    persist_if_new, is_compound_mood,
    groq_parse_refinement, franchise_key,
    get_era_range,
)

# ── Shared Groq client ────────────────────────────────────────────────────────
_groq_client = None
def _get_groq():
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client


# ── Valid genres ──────────────────────────────────────────────────────────────
VALID_GENRES = {
    'Action', 'Adventure', 'Animation', 'Comedy', 'Crime',
    'Documentary', 'Drama', 'Family', 'Fantasy', 'History',
    'Horror', 'Music', 'Mystery', 'Romance', 'Science Fiction',
    'Thriller', 'War', 'Western',
}

# ── Language maps ─────────────────────────────────────────────────────────────
_MOOD_LANG_HINTS = {
    'bollywood': 'hi', 'hindi': 'hi', 'hindi film': 'hi',
    'hindi movie': 'hi', 'desi': 'hi', 'indian movie': 'hi',
    'south indian': 'south_asian_regional', 'tollywood': 'te',
    'kollywood': 'ta', 'mollywood': 'ml',
    'tamil': 'ta', 'telugu': 'te', 'malayalam': 'ml', 'kannada': 'kn',
    'marathi': 'mr', 'bengali': 'bn', 'punjabi': 'pa',
    'korean': 'ko', 'k-drama': 'ko', 'kdrama': 'ko', 'k drama': 'ko',
    'japanese': 'ja', 'anime': 'ja',
    'chinese': 'zh', 'mandarin': 'zh',
    'hollywood': 'en', 'english': 'en', 'american': 'en', 'british': 'en',
    'french': 'fr', 'spanish': 'es', 'italian': 'it',
    'german': 'de', 'portuguese': 'pt',
}

_MOOD_LANG_GROUPS = {
    'hi': 'south_asian', 'ta': 'south_asian', 'te': 'south_asian',
    'ml': 'south_asian', 'kn': 'south_asian', 'bn': 'south_asian',
    'mr': 'south_asian', 'pa': 'south_asian', 'ur': 'south_asian',
    'en': 'western',     'fr': 'western',     'es': 'western',
    'it': 'western',     'de': 'western',     'pt': 'western',
    'zh': 'east_asian',  'ja': 'east_asian',  'ko': 'east_asian',
}


# ── Groq mood interpretation ──────────────────────────────────────────────────
def groq_interpret_mood(mood: str) -> dict:
    """
    Groq interprets any mood string and returns matching genres + vibe + keywords.
    Returns a confidence score so we know when to fall back to llm_chat.
    """
    prompt = (
        f"A user wants movie recommendations. Their mood/feeling is: '{mood}'.\n"
        f"Think carefully about what kind of movie ACTUALLY fits this feeling.\n\n"
        f"Return ONLY a JSON object, no explanation:\n"
        f"{{\n"
        f'  "genres": ["Genre1", "Genre2"],\n'
        f'  "exclude_genres": ["Genre3"],\n'
        f'  "vibe": "one sentence describing the ideal movie for this mood",\n'
        f'  "keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"],\n'
        f'  "confidence": 0.0\n'
        f"}}\n\n"
        f"Rules:\n"
        f"- genres: 1-3 genres that best match the mood. MUST be from this list:\n"
        f"  Action, Adventure, Animation, Comedy, Crime, Documentary, Drama, "
        f"Family, Fantasy, History, Horror, Music, Mystery, Romance, "
        f"Science Fiction, Thriller, War, Western\n"
        f"- exclude_genres: genres to explicitly AVOID (e.g. if user says 'not romance' → ['Romance'])\n"
        f"- keywords: SPECIFIC plot-level words found in movie overviews (NOT tone words).\n"
        f"  e.g. 'girls night' → ['friendship', 'female', 'women', 'bonding', 'fun']\n"
        f"  e.g. 'rainy sunday' → ['cozy', 'comfort', 'nostalgic', 'slow', 'heartwarming']\n"
        f"  e.g. 'heartbroken' → ['breakup', 'grief', 'loss', 'loneliness', 'healing']\n"
        f"- vibe: describe HOW the movie should feel, not what it's about\n"
        f"- confidence: 0.8+ if mood maps cleanly. <0.5 if very niche or adult/explicit.\n"
        f"- If mood mentions a specific director, actor, or year — confidence = 0.0"
    )
    for attempt in range(2):
        try:
            resp = _get_groq().chat.completions.create(
                model='llama-3.3-70b-versatile',
                messages=[{'role': 'user', 'content': prompt}],
                temperature=0.1,
                max_tokens=250,
            )
            raw = resp.choices[0].message.content.strip()
            raw = re.sub(r'^```json?\s*', '', raw, flags=re.MULTILINE)
            raw = re.sub(r'```\s*$',      '', raw, flags=re.MULTILINE)
            result = json.loads(raw.strip())
            result['genres'] = [g for g in result.get('genres', []) if g in VALID_GENRES]
            result['exclude_genres'] = [g for g in result.get('exclude_genres', []) if g in VALID_GENRES]
            result.setdefault('confidence', 0.8)
            result.setdefault('keywords', [])
            return result
        except Exception as e:
            err = str(e)
            if '429' in err and attempt == 0:
                m = re.search(r'try again in (\d+\.?\d*)s', err)
                wait = min(int(float(m.group(1))) + 1, 12) if m else 6
                print(f'[mood] Rate limit — waiting {wait}s')
                _time.sleep(wait)
            else:
                print(f'[Groq mood] Error: {e}')
                return None
    return None


# ── Groq mood ranker ──────────────────────────────────────────────────────────
def _groq_rank_mood(mood_str, vibe_str, candidates_df, top_n):
    """
    Groq picks the best emotional matches from the DB candidate pool.
    Returns list of positional indices into candidates_df.
    """
    if not GROQ_API_KEY or GROQ_API_KEY == 'YOUR_GROQ_KEY_HERE':
        return []

    lines = []
    for i, (_, row) in enumerate(candidates_df.iterrows(), 1):
        try:
            yr = int(row['year']) if row.get('year') is not None and not pd.isna(row['year']) else '?'
        except (TypeError, ValueError):
            yr = '?'
        va  = float(row.get('vote_average', 0) or 0)
        ov  = str(row.get('overview', '') or '')[:100].replace('\n', ' ')
        gen = str(row.get('genres', '') or '')
        lines.append(f'{i}. "{row["title"]}" ({yr}) | {gen} | ★{va:.1f} | {ov}')

    prompt = (
        f'A user wants movie recommendations for this mood: "{mood_str}"\n'
        f'The ideal movie should feel like: {vibe_str}\n\n'
        f'From this candidate list, pick the BEST {top_n} films.\n'
        f'RANKING CRITERIA (in order of importance):\n'
        f'1. Emotional fit — does the film actually match the mood/occasion?\n'
        f'2. Tone match — funny=light/fun, sad=emotional, thriller=tense\n'
        f'3. Quality — higher rated films preferred when vibe is equal\n'
        f'4. Prefer feature films unless the mood specifically asks for documentaries or stand-up specials\n'
        f'Return ONLY a JSON array of item numbers in ranked order, e.g. [3,7,1,9,5]\n'
        f'No explanation. No markdown.\n\n'
        + '\n'.join(lines)
    )

    for attempt in range(2):
        try:
            resp = _get_groq().chat.completions.create(
                model='llama-3.3-70b-versatile',
                messages=[{'role': 'user', 'content': prompt}],
                temperature=0.1,
                max_tokens=150,
            )
            raw = resp.choices[0].message.content.strip()
            raw = re.sub(r'[^0-9,\[\]]', '', raw)
            picks = json.loads(raw)
            picks = [p - 1 for p in picks if isinstance(p, int) and 1 <= p <= len(candidates_df)]
            print(f'[mood] Groq ranked: {picks[:top_n]}')
            return picks[:top_n]
        except Exception as e:
            err = str(e)
            if '429' in err and attempt == 0:
                m = re.search(r'try again in (\d+\.?\d*)s', err)
                wait = min(int(float(m.group(1))) + 1, 12) if m else 6
                print(f'[mood] Rate limit — waiting {wait}s')
                _time.sleep(wait)
            else:
                print(f'[mood] Groq rank failed: {err[:80]}')
                return []
    return []


# ── Main function ─────────────────────────────────────────────────────────────
def mood_based_recommend(mood: str, top_n: int = 10,
                         era: str = None, min_votes: int = 500,
                         refine: str = None,
                         exclude_titles: list = None):
    """
    Accepts ANY mood string — single word, phrase, or free text.
    Returns (results, error) or (None, 'LLM_FALLBACK') to signal agent
    to route to llm_chat instead.
    """
    global LAST_SESSION
    _size_before   = len(df)
    genres_exclude = []

    # ── Handle refine ─────────────────────────────────────────────────────────
    if refine:
        if not LAST_SESSION['query']:
            return None, "No previous session to refine."
        print(f"[mood] Refining: '{refine}'")
        parsed         = groq_parse_refinement(refine, LAST_SESSION)
        action         = parsed.get('action', 'more')
        genres_exclude = parsed.get('genres_exclude') or []
        if action == 'new_mood':
            mood = parsed.get('new_mood') or refine
            print(f"[mood] New mood: '{mood}'")
        else:
            mood = LAST_SESSION['query']
        exclude_titles = list(LAST_SESSION['shown']) + (exclude_titles or [])
        era            = era or LAST_SESSION.get('era')
        min_votes      = LAST_SESSION.get('min_votes', 500)
        top_n          = LAST_SESSION.get('top_n', top_n)
        print(f"[mood] Re-running mood='{mood}' excluding {len(exclude_titles)} shown titles")

    # ── Groq interprets mood ──────────────────────────────────────────────────
    print(f"[mood] Interpreting '{mood}'...")
    result = groq_interpret_mood(mood)
    if result is None:
        persist_if_new(_size_before)
        return None, f"Could not interpret '{mood}'. Check Groq API key."

    target_genres  = result.get('genres', [])
    exclude_genres = result.get('exclude_genres', []) + genres_exclude
    confidence     = result.get('confidence', 0.8)
    match_all      = is_compound_mood(mood)
    vibe           = result.get('vibe', '')
    keywords       = [k.lower() for k in result.get('keywords', [])]

    print(f"[mood] genres={target_genres}  exclude={exclude_genres}  confidence={confidence}")
    print(f"[mood] vibe: {vibe}")
    print(f"[mood] keywords: {keywords}")

    # ── Low confidence → LLM fallback ────────────────────────────────────────
    if confidence < 0.5 or not target_genres:
        print(f"[mood] Low confidence ({confidence}) — signalling LLM fallback")
        persist_if_new(_size_before)
        return None, 'LLM_FALLBACK'

    # ── Era filter ────────────────────────────────────────────────────────────
    year_min, year_max = get_era_range(era)
    if era:
        print(f"[mood] Era: {year_min}–{year_max}")
    else:
        print(f"[mood] Default era: 1970+")

    top2      = {g.lower() for g in target_genres[:2]}
    tnorm     = {g.lower() for g in target_genres}
    excl_norm = {g.lower() for g in exclude_genres}

    def matches(gl):
        if not isinstance(gl, list): return False
        mg = {g.lower().strip() for g in gl}
        if excl_norm and mg & excl_norm: return False
        return top2.issubset(mg) if match_all else bool(mg & tnorm)

    _vc = pd.to_numeric(df['vote_count'],   errors='coerce').fillna(0)
    _va = pd.to_numeric(df['vote_average'], errors='coerce').fillna(0)
    _yr = pd.to_numeric(df['year'],         errors='coerce').fillna(0)

    def build_mask(mv):
        m = df['genre_list'].apply(matches) & (_vc >= mv) & (_va >= 5.5)
        if year_min: m = m & (_yr >= year_min)
        if year_max: m = m & (_yr <= year_max)
        return m

    # ── Auto-relax min_votes if needed ────────────────────────────────────────
    results = pd.DataFrame()
    for threshold in [min_votes, 200, 50, 10]:
        results = df[build_mask(threshold)].copy()
        if len(results) >= top_n * 3:
            if threshold != min_votes:
                print(f"[mood] Relaxed min_votes {min_votes}→{threshold}")
            break

    if results.empty:
        print(f"[mood] No results after all thresholds — signalling LLM fallback")
        persist_if_new(_size_before)
        return None, 'LLM_FALLBACK'

    # ── Exclude shown titles ──────────────────────────────────────────────────
    if exclude_titles:
        excl    = {re.sub(r'[^a-z0-9 ]', '', t.lower().strip()) for t in exclude_titles}
        results = results[~results['title_clean'].isin(excl)]

    # ── Language filter ───────────────────────────────────────────────────────
    mood_lower    = mood.lower()
    detected_lang = None
    for hint, lang_code in _MOOD_LANG_HINTS.items():
        if hint in mood_lower:
            detected_lang = lang_code
            break

    if detected_lang is None and LAST_SESSION.get('query_lang'):
        detected_lang = LAST_SESSION['query_lang']
        print(f"[mood] No lang hint — using session lang: {detected_lang}")

    if 'original_language' in results.columns:
        lang_col  = results['original_language'].astype(str).str.lower().str.strip()
        min_needed = max(top_n // 2, 3)

        if detected_lang == 'south_asian_regional':
            filt = lang_col.isin(['ta', 'te', 'ml', 'kn'])
            if filt.sum() >= min_needed:
                results = results[filt]
                print(f'[mood] Lang filter: south_asian_regional ({filt.sum()})')

        elif detected_lang == 'en':
            filt = lang_col == 'en'
            if filt.sum() >= min_needed:
                results = results[filt]
                print(f'[mood] Lang filter: english only ({filt.sum()})')

        elif detected_lang:
            # Try exact language first
            filt_exact = lang_col == detected_lang
            if filt_exact.sum() >= min_needed:
                results = results[filt_exact]
                print(f'[mood] Lang filter: exact={detected_lang} ({filt_exact.sum()})')
            else:
                # Fall back to language + English
                filt = lang_col.isin([detected_lang, 'en'])
                if filt.sum() >= min_needed:
                    results = results[filt]
                    print(f'[mood] Lang filter: {detected_lang}+en ({filt.sum()})')
                else:
                    grp = _MOOD_LANG_GROUPS.get(detected_lang, 'other')
                    filt2 = lang_col.apply(lambda l: l == 'en' or _MOOD_LANG_GROUPS.get(l, 'other') == grp)
                    if filt2.sum() >= min_needed:
                        results = results[filt2]
                        print(f'[mood] Lang filter: group={grp} ({filt2.sum()})')

        print(f"[mood] Lang filter: detected={detected_lang} results={len(results)}")

    # ── Keyword boost scoring ─────────────────────────────────────────────────
    def keyword_boost(row):
        if not keywords:
            return 0.0
        text = ' '.join([
            str(row.get('title',       '') or ''),
            str(row.get('overview',    '') or ''),
            str(row.get('tagline',     '') or ''),
            str(row.get('keyword_str', '') or '') if 'keyword_str' in df.columns else '',
            str(row.get('keywords',    '') or ''),
        ]).lower()
        hits = sum(1 for kw in keywords if kw in text)
        return hits / len(keywords)

    # ── Specificity + quality + keyword scoring ───────────────────────────────
    def specificity(gl):
        if not isinstance(gl, list) or not gl: return 0.0
        mg    = {g.lower().strip() for g in gl}
        score = len(mg & tnorm) / len(mg)
        bonus = 0.2 if gl[0].lower().strip() in tnorm else 0.0
        return score + bonus

    results['spec']     = results['genre_list'].apply(specificity)
    results['kw_boost'] = results.apply(keyword_boost, axis=1)

    # Quality score
    try:
        _va2 = pd.to_numeric(results['vote_average'], errors='coerce').fillna(0)
        _vc2 = pd.to_numeric(results['vote_count'],   errors='coerce').fillna(1).clip(lower=1)
        proxy = _va2 * np.log10(_vc2)
        results['qual'] = (proxy - proxy.min()) / (proxy.max() - proxy.min() + 1e-9)
    except Exception:
        results['qual'] = 0.5

    # Language boost
    if detected_lang and detected_lang not in ('en', None):
        lang_col2 = results['original_language'].astype(str).str.lower().str.strip() \
                    if 'original_language' in results.columns else None
        if lang_col2 is not None:
            results['lang_boost'] = lang_col2.apply(
                lambda l: 0.3 if l == detected_lang else 0.0
            )
        else:
            results['lang_boost'] = 0.0
    else:
        results['lang_boost'] = 0.0

    results['final_score'] = (
        results['spec']       * 0.20 +
        results['qual']       * 0.40 +
        results['kw_boost']   * 0.20 +
        results['lang_boost'] * 0.20
    )
    results = results.sort_values('final_score', ascending=False)

    # ── Franchise dedup ───────────────────────────────────────────────────────
    results['_fr'] = results['title'].apply(franchise_key)
    results = results.drop_duplicates('_fr', keep='first').drop(columns='_fr')

    # ── Groq ranking — pick best emotional matches from top pool ─────────────
    top_pool = results.head(max(top_n * 3, 20))
    if GROQ_API_KEY and GROQ_API_KEY != 'YOUR_GROQ_KEY_HERE' and len(top_pool) >= top_n:
        picks = _groq_rank_mood(mood, vibe, top_pool, top_n)
        if picks:
            final = top_pool.iloc[picks].head(top_n)
            print(f'[mood] Groq selected {len(final)} films')
        else:
            final = results.head(top_n)
    else:
        final = results.head(top_n)

    # ── Save session state ────────────────────────────────────────────────────
    LAST_SESSION.update({
        'tool':       'mood',
        'query':      mood,
        'shown':      LAST_SESSION['shown'] + list(final['title'].values) if refine
                      else list(final['title'].values),
        'era':        era,
        'min_votes':  min_votes,
        'top_n':      top_n,
        'query_lang': detected_lang,
    })

    persist_if_new(_size_before)

    cols = ['title', 'year', 'genres', 'vote_average', 'vote_count', 'overview', 'director', 'poster_url']
    return final[[c for c in cols if c in final.columns]].to_dict('records'), None