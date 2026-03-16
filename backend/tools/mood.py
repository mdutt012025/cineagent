# backend/tools/mood.py
# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 2: mood_based_recommend
# ═══════════════════════════════════════════════════════════════════════════════

import re
import json
import pandas as pd
from groq import Groq

from .shared import (
    df, GROQ_API_KEY, LAST_SESSION,
    persist_if_new, is_compound_mood,
    groq_parse_refinement, franchise_key,
    get_era_range,
)

# Valid genres in dataset — Groq must pick from these only
VALID_GENRES = {
    'Action', 'Adventure', 'Animation', 'Comedy', 'Crime',
    'Documentary', 'Drama', 'Family', 'Fantasy', 'History',
    'Horror', 'Music', 'Mystery', 'Romance', 'Science Fiction',
    'Thriller', 'War', 'Western',
}


def groq_interpret_mood(mood: str) -> dict:
    """
    Groq interprets any mood string and returns matching genres + vibe + keywords.
    Also returns a confidence score so we know when to fall back to llm_chat.
    """
    prompt = (
        f"A user wants movie recommendations. Their mood/feeling is: '{mood}'.\n"
        f"Think carefully about what kind of movie ACTUALLY fits this feeling.\n\n"
        f"Return ONLY a JSON object, no explanation:\n"
        f"{{\n"
        f'  "genres": ["Genre1", "Genre2"],\n'
        f'  "vibe": "one sentence describing the ideal movie for this mood",\n'
        f'  "keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"],\n'
        f'  "confidence": 0.0\n'
        f"}}\n\n"
        f"Rules:\n"
        f"- confidence: 0.0-1.0. High (0.8+) if mood maps cleanly to genres. "
        f"Low (<0.5) if mood is very specific, niche, adult, or hard to map to genres.\n"
        f"- genres MUST only come from this exact list:\n"
        f"  Action, Adventure, Animation, Comedy, Crime, Documentary, Drama, "
        f"Family, Fantasy, History, Horror, Music, Mystery, Romance, "
        f"Science Fiction, Thriller, War, Western\n"
        f"- keywords should capture tone, themes, and feeling (e.g. 'passionate', "
        f"'heartwarming', 'suspenseful', 'steamy', 'dark', 'uplifting')\n"
        f"- If the mood is adult/explicit/sexual in nature, set confidence below 0.5\n"
        f"- If the mood mentions a specific director, actor, or year — confidence = 0.0"
    )
    import time as _time
    for attempt in range(2):
        try:
            resp = Groq(api_key=GROQ_API_KEY).chat.completions.create(
                model='llama-3.3-70b-versatile',
                messages=[{'role': 'user', 'content': prompt}],
                temperature=0.1,
                max_tokens=200,
            )
            raw = resp.choices[0].message.content.strip()
            raw = re.sub(r'^```json?\s*', '', raw, flags=re.MULTILINE)
            raw = re.sub(r'```\s*$',      '', raw, flags=re.MULTILINE)
            result = json.loads(raw.strip())
            result['genres'] = [g for g in result.get('genres', []) if g in VALID_GENRES]
            result.setdefault('confidence', 0.8)
            result.setdefault('keywords', [])
            return result
        except Exception as e:
            err = str(e)
            if '429' in err and attempt == 0:
                import re as _re2
                m = _re2.search(r'try again in (\d+\.?\d*)s', err)
                wait = min(int(float(m.group(1))) + 1, 12) if m else 6
                print(f'[mood] Rate limit — waiting {wait}s')
                _time.sleep(wait)
            else:
                print(f'[Groq mood] Error: {e}')
                return None
    return None


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
    _size_before = len(df)
    genres_exclude = []

    # ── Handle refine= ────────────────────────────────────────────────────────
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
        print(f"[mood] Re-running mood='{mood}' "
              f"excluding {len(exclude_titles)} shown titles")

    # ── Groq interprets mood ──────────────────────────────────────────────────
    print(f"[mood] Interpreting '{mood}'...")
    result = groq_interpret_mood(mood)
    if result is None:
        persist_if_new(_size_before)
        return None, f"Could not interpret '{mood}'. Check Groq API key."

    target_genres = result.get('genres', [])
    confidence    = result.get('confidence', 0.8)
    match_all     = is_compound_mood(mood)
    vibe          = result.get('vibe', '')
    keywords      = [k.lower() for k in result.get('keywords', [])]

    print(f"[mood] genres={target_genres}  confidence={confidence}  match_all={match_all}")
    print(f"[mood] vibe: {vibe}")
    print(f"[mood] keywords: {keywords}")

    # ── Low confidence → signal agent to use llm_chat ─────────────────────────
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
    excl_norm = {g.lower() for g in genres_exclude}

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
        persist_if_new(_size_before)
        # No genre matches at all — fall back to LLM
        print(f"[mood] No results after all thresholds — signalling LLM fallback")
        return None, 'LLM_FALLBACK'

    # ── Adult content guard ──────────────────────────────────────────────────
    _ADULT_KEYWORDS = {
        'sensual', 'erotic', 'sexy', 'steamy', 'adult', 'explicit',
        'sexual', 'nude', 'bold', 'hot scenes', 'nsfw', 'erotica',
    }
    mood_lower = mood.lower()
    if any(kw in mood_lower for kw in _ADULT_KEYWORDS):
        print("[mood] Adult keyword detected — sanitizing mood")
        mood = re.sub(
            r'\b(sensual|erotic|sexy|steamy|explicit|sexual|bold|hot|nude|nsfw)\b',
            'passionate', mood_lower
        ).strip()
        print(f"[mood] Sanitized mood: '{mood}'")

    # ── Exclude shown titles ──────────────────────────────────────────────────
    if exclude_titles:
        excl    = {re.sub(r'[^a-z0-9 ]', '', t.lower().strip()) for t in exclude_titles}
        results = results[~results['title_clean'].isin(excl)]

    # ── Language filter ───────────────────────────────────────────────────────
    _MOOD_LANG_HINTS = {
        'bollywood': 'hi', 'hindi': 'hi', 'desi': 'hi', 'indian movie': 'hi',
        'south indian': 'south_asian_regional',
        'tollywood': 'te', 'kollywood': 'ta', 'mollywood': 'ml',
        'tamil': 'ta', 'telugu': 'te', 'malayalam': 'ml', 'kannada': 'kn',
        'korean': 'ko', 'k-drama': 'ko', 'kdrama': 'ko',
        'japanese': 'ja', 'anime': 'ja',
        'chinese': 'zh', 'mandarin': 'zh',
        'hollywood': 'en', 'english': 'en', 'american': 'en', 'british': 'en',
        'french': 'fr', 'spanish': 'es', 'italian': 'it', 'german': 'de',
    }
    _MOOD_LANG_GROUPS = {
        'hi': 'south_asian', 'ta': 'south_asian', 'te': 'south_asian',
        'ml': 'south_asian', 'kn': 'south_asian', 'bn': 'south_asian',
        'en': 'western',     'fr': 'western',     'es': 'western',
        'it': 'western',     'de': 'western',     'pt': 'western',
        'zh': 'east_asian',  'ja': 'east_asian',  'ko': 'east_asian',
    }
    detected_lang = None
    for hint, lang_code in _MOOD_LANG_HINTS.items():
        if hint in mood_lower:
            detected_lang = lang_code
            break
    if detected_lang is None and LAST_SESSION.get('query_lang'):
        detected_lang = LAST_SESSION['query_lang']

    if 'original_language' in results.columns:
        lang_col = results['original_language'].astype(str).str.lower().str.strip()
        if detected_lang == 'south_asian_regional':
            filt = lang_col.isin(['ta', 'te', 'ml', 'kn'])
            if filt.sum() >= top_n:
                results = results[filt]
        elif detected_lang == 'en':
            filt = lang_col == 'en'
            if filt.sum() >= top_n:
                results = results[filt]
        elif detected_lang:
            filt = lang_col.isin([detected_lang, 'en'])
            if filt.sum() >= top_n:
                results = results[filt]
            else:
                grp = _MOOD_LANG_GROUPS.get(detected_lang, 'other')
                filt2 = lang_col.apply(lambda l: l == 'en' or _MOOD_LANG_GROUPS.get(l, 'other') == grp)
                if filt2.sum() >= top_n:
                    results = results[filt2]
        print(f"[mood] Lang filter: detected={detected_lang} results={len(results)}")

    # ── Keyword boost scoring ─────────────────────────────────────────────────
    def keyword_boost(row):
        """Score how well a movie's text matches the mood keywords."""
        if not keywords:
            return 0.0
        text = ' '.join([
            str(row.get('overview',     '') or ''),
            str(row.get('tagline',      '') or ''),
            str(row.get('keyword_str',  '') or ''),
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

    results['spec']    = results['genre_list'].apply(specificity)
    results['kw_boost'] = results.apply(keyword_boost, axis=1)

    if 'weighted_score' in results.columns:
        ws      = pd.to_numeric(results['weighted_score'], errors='coerce').fillna(0)
        ws_min  = ws.min()
        ws_max  = ws.max()
        results['qual'] = (ws - ws_min) / (ws_max - ws_min + 1e-9)
    else:
        # Fallback: use vote_average × log(vote_count) as quality proxy
        import numpy as _np
        _va2 = pd.to_numeric(results['vote_average'], errors='coerce').fillna(0)
        _vc2 = pd.to_numeric(results['vote_count'],   errors='coerce').fillna(1).clip(lower=1)
        proxy = _va2 * _np.log10(_vc2)
        results['qual'] = (proxy - proxy.min()) / (proxy.max() - proxy.min() + 1e-9)

    # Weights: specificity 25%, quality 50%, keyword boost 25%
    results['final_score'] = (
        results['spec']     * 0.25 +
        results['qual']     * 0.50 +
        results['kw_boost'] * 0.25
    )
    results = results.sort_values('final_score', ascending=False)

    # ── Franchise dedup ───────────────────────────────────────────────────────
    results['_fr'] = results['title'].apply(franchise_key)
    results = results.drop_duplicates('_fr', keep='first').drop(columns='_fr')

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