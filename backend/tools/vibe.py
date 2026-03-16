# backend/tools/vibe.py
# Clean rewrite — structured fingerprint approach, defensive coding throughout

import re
import unicodedata
import json
import math
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

from .shared import (
    df, tfidf_matrix, title_to_idx,
    GROQ_API_KEY, LAST_SESSION, _VIBE_CACHE,
    resolve, persist_if_new, check_disambiguation,
    groq_parse_refinement, get_era_range,
    tmdb_search, groq_enrich, inject_movie, safe_row,
    ARTIFACTS_DIR,
)

# Import embed_matrix — None if generate_embeddings.py hasn't been run yet
try:
    from .shared import embed_matrix, faiss_index
except ImportError:
    embed_matrix = None
    faiss_index  = None

try:
    from .shared import CURRENT_YEAR
except ImportError:
    CURRENT_YEAR = datetime.now().year

_MAX_SESSION_SHOWN = 50

# ── Genre compatibility ────────────────────────────────────────────────────────
# For each query genre, which candidate PRIMARY genres are hard blocks
_PRIMARY_GENRE_BLOCKS = {
    'romance':   {'action', 'horror', 'war', 'documentary', 'western', 'science fiction', 'thriller'},
    'comedy':    {'horror', 'war', 'documentary', 'western', 'history'},
    'drama':     {'horror', 'war', 'western'},
    'horror':    {'romance', 'animation', 'family', 'comedy'},
    'thriller':  {'animation', 'family'},
    'action':    {'documentary', 'horror'},
    'adventure': {'horror', 'documentary'},
    'family':    {'horror', 'thriller', 'war'},
}

# _VIBE_NEVER_MATCH removed — genre blocking and emotional axis scoring
# handle these cases generically without hardcoding specific titles.



def _normalize_title(title):
    """Strip accents then non-alphanumeric. Amélie->amelie, 3 Idiots->3 idiots."""
    try:
        nfd = unicodedata.normalize('NFD', str(title))
        asc = nfd.encode('ascii', 'ignore').decode('ascii')
    except Exception:
        asc = str(title)
    return re.sub(r'[^a-z0-9 ]', '', asc.lower().strip())

def _get_genres(row):
    """Robustly extract genre list from a row dict."""
    gl = row.get('genre_list') or []
    if isinstance(gl, list) and gl:
        return [g.lower().strip() for g in gl if g]
    if isinstance(gl, str) and gl.strip().startswith('['):
        try:
            import ast as _ast
            parsed = _ast.literal_eval(gl)
            if parsed:
                return [g.lower().strip() for g in parsed if g]
        except Exception:
            pass
    genres_str = str(row.get('genres', '') or '')
    if genres_str.strip():
        return [g.strip().lower() for g in genres_str.split(',') if g.strip()]
    return []


def _query_primary_genres(query_row):
    return set(_get_genres(query_row))


def _is_hard_genre_mismatch(query_genres, candidate_row):
    """Block candidates whose top-2 genres conflict with the query film."""
    if not query_genres:
        return False
    cg = _get_genres(candidate_row)
    if not cg:
        return False
    top2     = set(cg[:2])
    primary  = cg[0]
    for qg in query_genres:
        blocked = _PRIMARY_GENRE_BLOCKS.get(qg, set())
        if not blocked:
            continue
        # Block if top-2 are ALL blocked with no overlap with query genres
        if top2.issubset(blocked) and not (top2 & query_genres):
            return True
        # Block if primary is blocked and not a query genre
        if primary in blocked and primary not in query_genres:
            return True
    return False



def _safe_rating(row):
    """NaN-safe, string-safe vote_average extraction."""
    raw = row.get('vote_average')
    if raw is None:
        return 0.0
    try:
        val = float(str(raw).strip())
        return 0.0 if math.isnan(val) or math.isinf(val) else val
    except (TypeError, ValueError):
        return 0.0


def _passes_quality(row, era, query_genres=None):
    """True if the row passes rating + genre + never-match checks."""
    rating     = _safe_rating(row)
    vote_count = float(row.get('vote_count', 0) or 0)
    # No hard rating floor — weighted scoring naturally deprioritizes low-rated films.
    # Only block truly unrated/zero-vote films with suspicious quality signals.
    if rating > 0 and rating < 4.5 and float(row.get('vote_count', 0) or 0) > 500:
        return False, 'rating {:.1f} too low (established poor quality)'.format(rating)

    if query_genres and _is_hard_genre_mismatch(query_genres, row):
        return False, 'genre mismatch ({})'.format(row.get('genres', ''))

    return True, ''


def _passes_era(row, era):
    """True if row's year fits the era."""
    if not era:
        return True
    yr_min, yr_max = get_era_range(era)
    row_year = int(row.get('year', 0) or 0)
    if yr_min and row_year < yr_min:
        return False
    if yr_max and row_year > yr_max:
        return False
    return True


# ── Groq helpers ───────────────────────────────────────────────────────────────

import time as _time

# In-memory fingerprint cache — avoids calling Groq twice for the same film
# within a single server session. Key: normalized title.
_FP_CACHE: dict = {}



# Shared Groq client — instantiated once, reused across calls
_groq_client = None
def _get_groq_client():
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client

def _groq(messages, temperature=0.1, max_tokens=400, _retry=1, model='llama-3.3-70b-versatile'):
    """
    Single Groq call with 429 retry backoff. Returns text or None.
    _retry=1: one retry max to avoid long waits causing HTTP timeouts.
    On 429: waits the server-specified time (capped at 15s) then retries once.
    """
    client = _get_groq_client()
    for attempt in range(_retry + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            err = str(e)
            if '429' in err and attempt < _retry:
                import re as _re
                m = _re.search(r'try again in (\d+\.?\d*)s', err)
                wait = min(int(float(m.group(1))) + 1, 15) if m else 8
                print('[Groq] Rate limit — waiting {}s (attempt {}/{})'.format(
                    wait, attempt + 1, _retry))
                _time.sleep(wait)
            else:
                print('[Groq] Call failed: {}'.format(err[:120]))
                return None
    return None


def _parse_json_obj(text):
    """Extract first JSON object from text."""
    if not text:
        return None
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text).strip()
    m = re.search(r'\{[\s\S]*\}', text)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return None


def _parse_json_arr(text):
    """Extract first JSON array from text."""
    if not text:
        return []
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text).strip()
    m = re.search(r'\[[\s\S]*\]', text)
    if not m:
        return []
    try:
        result = json.loads(m.group(0))
        return result if isinstance(result, list) else []
    except Exception:
        # Try to recover truncated JSON
        arr = m.group(0)
        last = arr.rfind('",')
        if last > 0:
            try:
                return json.loads(arr[:last+1] + ']')
            except Exception:
                pass
    return []


# ── Combined fingerprint + suggest (single Groq call) ─────────────────────────

# ── Emotional Axis Mega-Prompt ────────────────────────────────────────────────
# Single Groq call: fingerprint + suggest + filter + penalty — no round-trips.

_MEGA_PROMPT = """You are an emotional film similarity engine.
You must dynamically adapt to the emotional signature of the reference film.

Emotional axes (0-10 scale):
1. Psychological Complexity  - 0=surface/simple, 10=layered/ambiguous psychology
2. Darkness                  - 0=light/optimistic, 10=heavy/tragic/existential
3. Emotional Warmth          - 0=cold/detached, 10=tender/affectionate
4. Reality Instability       - 0=grounded realism, 10=dreams/distortion/ambiguity
5. Emotional Intensity       - 0=calm/subtle, 10=overwhelming/cathartic
6. Narrative Scope           - 0=intimate/personal, 10=epic/grand adventure

---
Reference Film: "{movie_title}" ({year})
Language: {lang}
Overview: {overview}
Known emotionally similar films: {refs}
Era constraint: {era}

STEP 1 - Profile the reference film on all 6 axes.

STEP 2 - Determine dominant emotional traits:
         - Identify the top 2 HIGHEST axes (dominant emotional drivers)
         - Identify the lowest 2 axes (emotional absences)
         - Determine tone direction: optimistic / dark / mixed

STEP 3 - Generate exactly {n} candidate films that align with:
         - Similar top emotional traits
         - Similar tone direction
         - Similar narrative scope
         LANGUAGE RULE (STRICT - HARD REQUIREMENT):
           - Reference language is {lang}.
           - If {lang} is Hindi/Bollywood, Tamil, Telugu, Malayalam or any Indian language:
               * AT LEAST 4 out of 5 films MUST be Indian films (Hindi, Tamil, Telugu, Malayalam etc.)
               * Maximum 1 English film allowed.
               * ZERO Korean, Japanese, French, Spanish, Thai or other non-Indian films.
           - If {lang} is Korean, Japanese, Chinese:
               * AT LEAST 3 out of 5 films MUST be in {lang}.
               * Rest can be English. ZERO Indian/Hindi films.
           - If {lang} is English:
               * English films only.
           - VIOLATING THIS RULE IS A HARD ERROR. Check every suggestion before returning.
         IMDB {min_rating}+. {era}.
         Do NOT include: {exclude}
         DO NOT default to European art films or Hollywood blockbusters.
         DO NOT use genre alone as criteria.

STEP 4 - For each candidate:
         a) Assign 6 axis scores
         b) emotional_distance = sum of |axis_i - ref_axis_i|
         c) REJECT the candidate if:
            - Any single axis differs by more than 5
            - Tone direction (optimistic vs dark) mismatches strongly
            - Candidate is primarily spectacle-driven: +3 penalty
            - Candidate is emotionally shallow vs reference: +2 penalty
            - Reference Emotional Warmth > 7 and candidate is violent/crime/thriller: +3 penalty

STEP 5 - Keep ONLY the 5 films with the LOWEST emotional_distance.
         FALLBACK: if fewer than 5 survive strict filtering, relax single-axis
         limit from 5 to 6 and fill to 5. Never return fewer than 5.

Return ONLY this JSON. No markdown. No explanation outside JSON:
{{
  "reference_profile": {{
    "psychological_complexity": 0,
    "darkness": 0,
    "emotional_warmth": 0,
    "reality_instability": 0,
    "emotional_intensity": 0,
    "narrative_scope": 0,
    "tone_direction": "optimistic|dark|mixed",
    "dominant_axes": ["axis1", "axis2"],
    "absent_axes": ["axis3", "axis4"]
  }},
  "recommendations": [
    {{
      "title": "Film Title",
      "year": 2019,
      "emotional_distance": 7,
      "why_match": "one sentence: shared emotional architecture, not plot"
    }}
  ]
}}
recommendations must contain exactly 5 items. JSON only. Be strict."""



def groq_suggest_and_score(query_title, query_row, n=20, exclude_titles=None, era=None):
    """
    Single Groq call: emotionally fingerprints the query film, generates candidates,
    scores each on 6 axes, applies warmth-based penalty, and returns only
    films with emotional_distance <= threshold. No separate filter call needed.
    """
    if not GROQ_API_KEY or GROQ_API_KEY == 'YOUR_GROQ_KEY_HERE':
        return []

    exclude_str = ', '.join((exclude_titles or [])[:15]) or 'none'
    lang        = str(query_row.get('original_language') or 'en').strip() or 'en'
    year        = int(query_row.get('year', 0)) if query_row.get('year') else ''
    genre_list  = _get_genres(query_row)
    genres      = ', '.join(genre_list) if genre_list else ''
    safe_ov     = str(query_row.get('overview', '') or '')[:250].replace('{', '(').replace('}', ')')

    lang_label = {
        'hi': 'Hindi/Bollywood', 'ta': 'Tamil', 'te': 'Telugu', 'ml': 'Malayalam',
        'ko': 'Korean', 'ja': 'Japanese', 'fr': 'French', 'es': 'Spanish',
        'de': 'German', 'it': 'Italian', 'zh': 'Chinese', 'en': 'English',
    }.get(lang, lang.upper() if lang else 'Unknown')

    # TF-IDF anchors — famous same-language neighbours for grounding
    qc         = re.sub(r'[^a-z0-9 ]', '', query_title.lower().strip())
    matrix_pos = title_to_idx.get(qc)
    ref_titles = []
    if matrix_pos is not None:
        sim    = cosine_similarity(tfidf_matrix[matrix_pos], tfidf_matrix).flatten()
        top200 = sim.argsort()[::-1][1:200]
        famous = sorted(
            [i for i in top200
             if df.iloc[i].to_dict().get('original_language') == lang
             and df.iloc[i].to_dict().get('vote_count', 0) > 10000],
            key=lambda i: df.iloc[i].to_dict().get('vote_count', 0), reverse=True
        )
        ref_titles = [df.iloc[i]['title'] for i in famous[:5]]
    ref_str = ', '.join(ref_titles) if ref_titles else 'none'

    # Era rule
    _era_str = {
        'new':     '2022-{}'.format(CURRENT_YEAR), 'modern': '2018-2021',
        '2010s':   '2010-2017', '2000s':   '2000-2009',
        '90s':     '1990-1999', 'classic':  '1940-1989',
    }
    era_rule = (
        'STRICT: Only films from {}. Reject anything outside.'.format(_era_str[era.lower()])
        if era and era.lower() in _era_str
        else 'Prefer {}-{}.'.format(max(1975, int(year) - 15), CURRENT_YEAR) if year
        else 'Any era, any decade.'
    )

    prompt = _MEGA_PROMPT.format(
        movie_title = query_row.get('title', query_title).replace('{', '(').replace('}', ')'),
        year        = year,
        lang        = lang_label,
        overview    = safe_ov,
        refs        = ref_str,
        era         = era_rule,
        n           = n + 10,
        min_rating  = 6.5,
        exclude     = exclude_str,
    )

    raw = _groq(
        [{'role': 'system', 'content': 'Return only valid JSON. No markdown. No explanation.'},
         {'role': 'user',   'content': prompt}],
        temperature = 0.3,
        max_tokens  = 800,   # reduced to stay under 6000 TPM
    )

    # Parse structured response: {reference_profile: {...}, recommendations: [...]}
    result = _parse_json_obj(raw)
    if result and isinstance(result.get('recommendations'), list):
        recs = result['recommendations']
        # Log reference profile
        ref = result.get('reference_profile', {})
        if ref:
            print('[axes] ref: pc={} d={} wa={} ri={} i={} ns={} tone={} dominant={}'.format(
                ref.get('psychological_complexity','?'),
                ref.get('darkness','?'),
                ref.get('emotional_warmth','?'),
                ref.get('reality_instability','?'),
                ref.get('emotional_intensity','?'),
                ref.get('narrative_scope','?'),
                ref.get('tone_direction','?'),
                ref.get('dominant_axes','?'),
            ))
    else:
        # Fallback: try plain array
        recs = _parse_json_arr(raw)

    if not recs:
        print('[Groq suggest] No recommendations returned')
        return []

    # Extract titles and log
    titles = []
    for f in recs:
        if not isinstance(f, dict):
            continue
        t    = str(f.get('title', '')).strip()
        dist = f.get('emotional_distance', '?')
        why  = str(f.get('why_match') or f.get('match_reason') or f.get('why_it_matches') or '')
        if t:
            titles.append(t)
            print('[axes] "{}" dist={} — {}'.format(t, dist, why[:65]))

    print('[Groq suggest] {} emotionally matched titles'.format(len(titles)))
    return titles

# ── Groq vibe filter ───────────────────────────────────────────────────────────

# ── Emotional Axis System ─────────────────────────────────────────────────────
# Films are compared in 6-dimensional emotional space, not by genre labels.
# Distance threshold: if sum of axis differences > AXIS_REJECT_THRESHOLD, reject.

AXIS_REJECT_THRESHOLD = 18  # max tolerable emotional distance (sum of 6 axis diffs)

_AXIS_FINGERPRINT_PROMPT = """You are an emotional film intelligence engine.
You do NOT classify by genre. You analyze films based on emotional structure.

Evaluate this film on 6 emotional axes (0-10 each):
1. Psychological Complexity  - 0=simple/surface, 10=layered/ambiguous psychology
2. Darkness                  - 0=light/optimistic, 10=heavy/tragic/existential
3. Emotional Warmth          - 0=cold/detached, 10=tender/affectionate/joyful
4. Reality Instability       - 0=grounded realism, 10=dreams/distortion/ambiguity
5. Emotional Intensity       - 0=calm/subtle, 10=overwhelming/cathartic
6. Narrative Scope           - 0=intimate/personal, 10=epic/grand adventure

CALIBRATION EXAMPLES (use these to anchor your scores):

Yeh Jawaani Hai Deewani — warm playful romantic Bollywood, youthful energy:
  pc=3, darkness=1, warmth=9, ri=2, intensity=6, scope=7, tone=optimistic
  dominant: [emotional_warmth, narrative_scope]

Dilwale Dulhania Le Jayenge — classic romantic Bollywood, emotional and warm:
  pc=3, darkness=2, warmth=9, ri=1, intensity=7, scope=6, tone=optimistic
  dominant: [emotional_warmth, emotional_intensity]

Parasite — dark class-war thriller, tense and suffocating:
  pc=8, darkness=8, warmth=2, ri=4, intensity=9, scope=5, tone=dark
  dominant: [darkness, psychological_complexity]

Amelie — whimsical romantic French film, dreamy and imaginative:
  pc=5, darkness=2, warmth=8, ri=7, intensity=5, scope=4, tone=optimistic
  dominant: [reality_instability, emotional_warmth]

Eternal Sunshine of the Spotless Mind — bittersweet surreal romance:
  pc=7, darkness=6, warmth=6, ri=9, intensity=8, scope=3, tone=mixed
  dominant: [reality_instability, emotional_intensity]

Requiem for a Dream — harrowing addiction descent, dark and spiraling:
  pc=6, darkness=10, warmth=1, ri=5, intensity=10, scope=3, tone=dark
  dominant: [darkness, emotional_intensity]

3 Idiots — warm comedic friendship film, hopeful and lighthearted:
  pc=4, darkness=2, warmth=8, ri=2, intensity=5, scope=5, tone=optimistic
  dominant: [emotional_warmth, narrative_scope]


Film: {movie_title}

Return ONLY valid JSON:
{open}
  "psychological_complexity": 0,
  "darkness": 0,
  "emotional_warmth": 0,
  "reality_instability": 0,
  "emotional_intensity": 0,
  "narrative_scope": 0,
  "tone_direction": "optimistic|dark|mixed",
  "dominant_axes": ["axis1", "axis2"],
  "core_emotional_experience": "one sentence: how the viewer feels after watching"
{close}
No explanation. No genre. JSON only."""

def get_emotional_fingerprint(movie_title, overview=''):
    """
    Score a film on 6 emotional axes.
    Cached in _FP_CACHE — same film never fingerprinted twice in a session.
    Returns validated dict, or tone-appropriate defaults on failure.
    """
    _AXIS_KEYS = ('psychological_complexity','darkness','emotional_warmth',
                  'reality_instability','emotional_intensity','narrative_scope')

    cache_key = re.sub(r'[^a-z0-9]', '', str(movie_title).lower())
    if cache_key in _FP_CACHE:
        fp = _FP_CACHE[cache_key]
        print('[axes] {} (cached) -> pc={} d={} wa={} ri={} i={} ns={} tone={}'.format(
            movie_title,
            fp.get('psychological_complexity','?'), fp.get('darkness','?'),
            fp.get('emotional_warmth','?'),         fp.get('reality_instability','?'),
            fp.get('emotional_intensity','?'),       fp.get('narrative_scope','?'),
            fp.get('tone_direction','?')))
        return fp

    safe_title = str(movie_title).replace('{','(').replace('}',')')
    prompt     = _AXIS_FINGERPRINT_PROMPT.format(
        movie_title=safe_title, open='{', close='}')
    raw = _groq([{'role': 'user', 'content': prompt}], temperature=0.1, max_tokens=250)
    fp  = _parse_json_obj(raw)
    valid = False

    if fp:
        axes = {k: int(fp.get(k, 5) or 5) for k in _AXIS_KEYS}
        if sum(axes.values()) > 5:   # reject near-zero rate-limit failures
            fp.update(axes)
            # Auto-correct: low darkness + high warmth = optimistic, not mixed
            if axes['darkness'] <= 3 and axes['emotional_warmth'] >= 7:
                if str(fp.get('tone_direction', '')).lower() == 'mixed':
                    fp['tone_direction'] = 'optimistic'
                    print('[axes] {} tone corrected mixed->optimistic'.format(movie_title))
            valid = True
        else:
            print('[axes] WARNING: near-zero axes for "{}" (sum={}) — using defaults'.format(
                movie_title, sum(axes.values())))

    if not valid:
        # Fallback based on overview keywords
        tone_guess = 'optimistic'
        if overview:
            ov = overview.lower()
            if any(w in ov for w in ['murder','kill','horror','terror','revenge','war','trauma']):
                tone_guess = 'dark'
        _defaults = {
            'optimistic': dict(psychological_complexity=4, darkness=2, emotional_warmth=8,
                               reality_instability=2, emotional_intensity=6, narrative_scope=6,
                               tone_direction='optimistic'),
            'dark':       dict(psychological_complexity=7, darkness=8, emotional_warmth=2,
                               reality_instability=4, emotional_intensity=8, narrative_scope=5,
                               tone_direction='dark'),
        }
        fp = dict(_defaults.get(tone_guess, dict(psychological_complexity=5, darkness=4,
                  emotional_warmth=5, reality_instability=3, emotional_intensity=6,
                  narrative_scope=5, tone_direction='mixed')))

    print('[axes] {} -> pc={} d={} wa={} ri={} i={} ns={} tone={}'.format(
        movie_title,
        fp.get('psychological_complexity','?'), fp.get('darkness','?'),
        fp.get('emotional_warmth','?'),         fp.get('reality_instability','?'),
        fp.get('emotional_intensity','?'),       fp.get('narrative_scope','?'),
        fp.get('tone_direction','?')))

    _FP_CACHE[cache_key] = fp
    return fp


# ── Candidate Pool Builder ─────────────────────────────────────────────────────
# Uses sentence-transformer embeddings when available, falls back to TF-IDF.
# Applies weighted scoring: 0.45×similarity + 0.20×vibe + 0.15×rating +
#                           0.10×popularity + 0.10×genre_overlap

def _genre_overlap_score(row, query_genres):
    """0.0–1.0 fraction of query genres present in candidate."""
    if not query_genres:
        return 0.5
    try:
        cand_genres = set(g.lower() for g in (_get_genres(row) or []))
        q_set       = set(g.lower() for g in query_genres)
        overlap     = len(q_set & cand_genres)
        return min(1.0, overlap / max(len(q_set), 1))
    except Exception:
        return 0.0


def _vibe_score(query_row, cand_row):
    """
    Cosine similarity of 7-dim emotional vibe vectors.
    Axes: intensity, warmth, darkness, romance, humor, hopefulness, scope
    Values derived from vote_average and genre heuristics — no LLM call needed.
    """
    def _vec(row):
        va      = float(row.get('vote_average', 5) or 5)
        genres  = set(g.lower() for g in (_get_genres(row) or []))
        dark    = 1.0 if any(g in genres for g in ['horror','thriller','crime','war']) else 0.0
        warm    = 1.0 if any(g in genres for g in ['romance','family','animation']) else 0.0
        humor   = 1.0 if 'comedy' in genres else 0.0
        scope   = 1.0 if any(g in genres for g in ['action','adventure','sci-fi','science fiction']) else 0.0
        intense = min(1.0, va / 10.0)
        hope    = max(0.0, 1.0 - dark)
        romance = warm * 0.8 + humor * 0.1
        return np.array([intense, warm, dark, romance, humor, hope, scope], dtype=np.float32)

    v1 = _vec(query_row)
    v2 = _vec(cand_row)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm == 0:
        return 0.5
    return float(np.dot(v1, v2) / norm)


def _era_score(row_year, era):
    """
    Returns 0.0–1.0 based on how well row_year matches the selected era.
    Perfect match = 1.0. Outside era = penalised by distance in years.
    """
    if not era:
        return 1.0
    yr_min, yr_max = _ERA_RANGES.get(str(era).lower(), (None, None))
    if yr_min is None and yr_max is None:
        return 1.0
    yr_min = yr_min or 0
    yr_max = yr_max or 9999
    year   = int(row_year or 0)
    if yr_min <= year <= yr_max:
        return 1.0
    # Penalise by how far outside the era the film is (decay over 10 years)
    dist = min(abs(year - yr_min), abs(year - yr_max))
    return max(0.0, 1.0 - dist / 10.0)


# Language affinity groups — films feel "natural" within the same cinematic culture
# Group A: South Asian (Bollywood, Tamil, Telugu, Malayalam, Bengali...)
# Group B: Western European (English, French, Spanish, Italian, German, Portuguese)
# Group C: East Asian (Chinese, Japanese, Korean)
# Group D: Everything else
_LANG_GROUPS = {
    'hi': 'south_asian', 'ta': 'south_asian', 'te': 'south_asian',
    'ml': 'south_asian', 'kn': 'south_asian', 'bn': 'south_asian',
    'mr': 'south_asian', 'pa': 'south_asian', 'ur': 'south_asian',
    'en': 'western',     'fr': 'western',     'es': 'western',
    'it': 'western',     'de': 'western',     'pt': 'western',
    'nl': 'western',     'sv': 'western',     'da': 'western',
    'zh': 'east_asian',  'ja': 'east_asian',  'ko': 'east_asian',
}

def _lang_affinity_score(query_lang, candidate_lang):
    """
    Language affinity — 3 tiers:
      1.0  exact same language       (hi→hi, ja→ja)
      0.9  English (always welcome)
      0.75 same group, diff language (hi→ta, hi→te — South Asian but not Hindi)
      0.85 adjacent group            (hi→en/fr, en→hi)
      0.5  cross-group               (hi→ja, ja→hi — culturally jarring)

    Key change: same-group ≠ same-language.
    Tamil/Telugu score 0.75 for a Hindi query, not 1.0.
    This stops South Indian films dominating Bollywood recommendations.
    """
    q = str(query_lang or 'en').lower().strip()
    c = str(candidate_lang or 'en').lower().strip()

    if c == q:
        return 1.0
    if c == 'en':
        return 0.9                    # English always acceptable
    q_group = _LANG_GROUPS.get(q, 'other')
    c_group = _LANG_GROUPS.get(c, 'other')
    if q_group == c_group:
        return 0.75                   # same region, different language — OK but not ideal
    if q_group == 'south_asian' and c_group == 'western':
        return 0.85                   # Bollywood fans enjoy English/French
    if q_group == 'western' and c_group == 'south_asian':
        return 0.7
    if q_group == 'east_asian' and c_group == 'western':
        return 0.7
    if q_group == 'western' and c_group == 'east_asian':
        return 0.65
    return 0.5                        # full cross-group penalty


# Era year ranges — single source of truth used by all era-aware functions
_ERA_RANGES = {
    'classic': (None, 1989),
    '90s':     (1990, 1999),
    '2000s':   (2000, 2009),
    '2010s':   (2010, 2017),
    'modern':  (2018, 2021),
    'new':     (2022, 9999),
}


def _weighted_score(sim, vibe, rating, vote_count, genre_overlap, era_s=1.0, lang_s=1.0,
                    native_boost=False):
    """
    Final ranking score:
    Default:      0.30×sim + 0.20×vibe + 0.15×rating + 0.10×pop + 0.10×genre + 0.05×era + 0.10×lang
    native_boost: 0.22×sim + 0.20×vibe + 0.15×rating + 0.10×pop + 0.10×genre + 0.05×era + 0.18×lang
    
    native_boost=True when query_lang is non-English — raises lang weight from 0.10→0.18
    so exact-language films rank meaningfully above cross-language ones.
    era_s=0.0  for out-of-era films       → strong penalty
    lang_s=0.75 for same-group diff lang   → prefers exact language match
    lang_s=0.5  for cross-group            → strong penalty (ja/zh for Hindi)
    """
    rating_norm = min(1.0, float(rating or 0) / 10.0)
    pop_norm    = min(1.0, math.log1p(float(vote_count or 0)) / math.log1p(1_000_000))
    if native_boost:
        return (0.22 * sim +
                0.20 * vibe +
                0.15 * rating_norm +
                0.10 * pop_norm +
                0.10 * genre_overlap +
                0.05 * era_s +
                0.18 * lang_s)
    return (0.30 * sim +
            0.20 * vibe +
            0.15 * rating_norm +
            0.10 * pop_norm +
            0.10 * genre_overlap +
            0.05 * era_s +
            0.10 * lang_s)


def _mmr(selected_rows, candidate_rows, embed_matrix_local, title_to_pos,
         lambda_=0.6, top_n=5):
    """
    Maximal Marginal Relevance — diversifies final results.
    lambda_=0.6 → 60% relevance, 40% diversity.
    Falls back gracefully if embeddings unavailable.
    """
    if embed_matrix_local is None or not candidate_rows:
        return candidate_rows[:top_n]

    def _get_vec(row):
        key = re.sub(r'[^a-z0-9 ]', '', str(row.get('title','')).lower().strip())
        pos = title_to_pos.get(key)
        if pos is None:
            return None
        return embed_matrix_local[pos]

    results    = list(selected_rows)
    remaining  = list(candidate_rows)
    used_keys  = {re.sub(r'[^a-z0-9 ]', '', str(r.get('title','')).lower().strip())
                  for r in results}

    while len(results) < top_n and remaining:
        best_score = -1
        best_row   = None
        best_idx   = 0

        for i, row in enumerate(remaining):
            key = re.sub(r'[^a-z0-9 ]', '', str(row.get('title','')).lower().strip())
            if key in used_keys:
                continue
            v = _get_vec(row)
            # Relevance: pre-computed weighted score stored in row
            relevance = float(row.get('_score', 0))
            # Diversity: max similarity to already-selected
            if results and v is not None:
                sel_vecs = [_get_vec(r) for r in results if _get_vec(r) is not None]
                if sel_vecs:
                    sims   = [float(np.dot(v, sv) /
                               max(np.linalg.norm(v) * np.linalg.norm(sv), 1e-8))
                              for sv in sel_vecs]
                    max_sim = max(sims)
                else:
                    max_sim = 0.0
            else:
                max_sim = 0.0

            mmr_score = lambda_ * relevance - (1 - lambda_) * max_sim
            if mmr_score > best_score:
                best_score = mmr_score
                best_row   = row
                best_idx   = i

        if best_row is None:
            break
        results.append(best_row)
        used_keys.add(re.sub(r'[^a-z0-9 ]', '', str(best_row.get('title','')).lower().strip()))
        remaining.pop(best_idx)

    return results[:top_n]


def build_candidate_pool(query_idx, query_clean, query_row, exclude_titles,
                         era, genres_exclude, genres_require,
                         top_n, query_genres=None, query_lang='en'):
    """
    Build a scored candidate pool of ~100 films.

    Strategy:
    1. Use embed_matrix (semantic) if available, else TF-IDF (keyword)
    2. Retrieve top 1000 by similarity
    3. Apply quality + era + genre filters
    4. Score each candidate with weighted formula
    5. Return top_n * 3 sorted by score (for Groq to rerank + MMR)
    """
    qr      = safe_row(query_idx)
    q_clean = re.sub(r'[^a-z0-9 ]', '', str(qr.get('title', '')).lower().strip())
    excl    = {re.sub(r'[^a-z0-9 ]', '', t.lower().strip()) for t in (exclude_titles or [])}

    # ── Step 1: similarity scores — FAISS > numpy embeddings > TF-IDF ─────────
    matrix_pos = title_to_idx.get(q_clean) or title_to_idx.get(query_clean)
    sim_scores = None
    top_idx    = None

    if faiss_index is not None and matrix_pos is not None:
        # FAISS path: IndexFlatIP returns cosine similarity directly (vectors pre-normalized)
        try:
            q_vec = embed_matrix[matrix_pos].reshape(1, -1).astype('float32') if embed_matrix is not None                     else None
            if q_vec is None:
                # embed_matrix not loaded but FAISS is — load just this vector
                import numpy as _np2
                _em   = _np2.load(str(ARTIFACTS_DIR / 'embed_matrix.npy'), mmap_mode='r')
                q_vec = _em[matrix_pos].reshape(1, -1).astype('float32')
            scores, indices = faiss_index.search(q_vec, 1001)
            top_idx    = indices[0]                  # shape (1001,)
            sim_scores_arr = scores[0]               # cosine similarities
            # Build full-length sim_scores array aligned to df index
            sim_scores = np.zeros(len(df), dtype=np.float32)
            valid = (top_idx >= 0) & (top_idx < len(df))
            sim_scores[top_idx[valid]] = sim_scores_arr[valid]
            top_idx = top_idx[valid][1:]             # skip self (rank 0)
            print('[pool] FAISS search: top {} candidates'.format(len(top_idx)))
        except Exception as e:
            print('[pool] FAISS search failed: {} — falling back'.format(e))
            sim_scores = None
            top_idx    = None

    if sim_scores is None and embed_matrix is not None and matrix_pos is not None:
        # Numpy cosine path
        q_vec      = embed_matrix[matrix_pos].reshape(1, -1)
        sim_scores = cosine_similarity(q_vec, embed_matrix).flatten()
        top_idx    = np.argsort(sim_scores)[::-1][1:1001]
        print('[pool] Numpy embedding search')

    if sim_scores is None:
        # TF-IDF fallback
        if matrix_pos is None:
            return pd.DataFrame()
        sim_scores = cosine_similarity(tfidf_matrix[matrix_pos], tfidf_matrix).flatten()
        top_idx    = np.argsort(sim_scores)[::-1][1:1001]
        print('[pool] TF-IDF search (run generate_embeddings.py + build_faiss_index.py for better results)')

    # ── Step 2: top 500 candidates ───────────────────────────────────────────
    top_idx_clean  = top_idx[:1000]
    candidates     = df.iloc[top_idx_clean].copy()
    candidates['_sim'] = sim_scores[top_idx_clean]
    candidates     = candidates[candidates['title_clean'] != q_clean]
    if excl:
        candidates = candidates[~candidates['title_clean'].isin(excl)]

    # ── Step 3a: language filter FIRST (from full 1000 pool) ────────────────
    # Strategy: ALWAYS try Hindi-first for south_asian queries, English as
    # universal fallback. Never fall back to random Korean/Thai/etc.
    #
    # Tier 1: exact same language (hi→hi, ko→ko, en→en)
    # Tier 2: same group (hi→hi/ta/te/ml/kn)
    # Tier 3: same language + English (hi+en, ko+en)
    # Tier 4 (FINAL fallback): Hindi + English  ← always safe, never random
    if 'original_language' in candidates.columns:
        ql       = str(query_lang).lower().strip() if query_lang else 'hi'
        ql_group = _LANG_GROUPS.get(ql, 'other')
        lang_col = candidates['original_language'].astype(str).str.lower().str.strip()

        # Tier 1: exact same language
        same_lang = candidates[lang_col == ql]
        if len(same_lang) >= top_n:
            candidates = same_lang
            print('[pool] Lang tier1: exact={} ({})'.format(len(candidates), ql))

        else:
            # Tier 2: same language group (south_asian/western/east_asian)
            group_only = candidates[lang_col.apply(
                lambda l: _LANG_GROUPS.get(l, 'other') == ql_group)]
            if len(group_only) >= top_n:
                candidates = group_only
                print('[pool] Lang tier2: group={} ({})'.format(ql_group, len(candidates)))

            else:
                # Tier 3: same language + English
                lang_en = candidates[lang_col.apply(
                    lambda l: l == ql or l == 'en')]
                if len(lang_en) >= top_n:
                    candidates = lang_en
                    print('[pool] Lang tier3: {}+en={}'.format(ql, len(candidates)))

                else:
                    # Tier 4: query language + English — universal safe fallback
                    # Never fall back to random Korean/Thai/French etc.
                    ql_en = candidates[lang_col.isin([ql, 'en'])]
                    if len(ql_en) >= top_n:
                        candidates = ql_en
                        print('[pool] Lang tier4: {}+en fallback ({})'.format(ql, len(candidates)))
                    else:
                        # Last resort: inject query language + English from full df
                        full_lang = df['original_language'].astype(str).str.lower().str.strip()
                        ql_en_df  = df[full_lang.isin([ql, 'en'])].copy()
                        ql_en_df['_sim'] = 0.2
                        candidates = pd.concat([candidates, ql_en_df]).drop_duplicates('title_clean')
                        print('[pool] Lang tier4: injected {}+en from df ({})'.format(ql, len(candidates)))

    # ── Step 3b: era filter (on language-filtered set) ────────────────────────
    # IMPORTANT: FAISS neighbors of a 2013 film are mostly 2010-2018 films.
    # If user asks for "New" era (2022+), almost none will be in the pool.
    # Solution: when in-era count is low, inject era-correct films from full df.
    if era:
        yr_min, yr_max = get_era_range(era)
        years = pd.to_numeric(candidates['year'], errors='coerce').fillna(0)
        era_mask = pd.Series([True] * len(candidates), index=candidates.index)
        if yr_min: era_mask &= years >= yr_min
        if yr_max: era_mask &= years <= yr_max
        era_count = int(era_mask.sum())

        if era_count >= top_n * 3:
            # Plenty of era matches — use only those
            candidates = candidates[era_mask]
            print('[pool] Era filter: {} in-era candidates'.format(era_count))
        else:
            # Not enough era matches in FAISS neighbors — inject from full df
            print('[pool] Era filter: only {} in FAISS pool, injecting from full df'.format(era_count))
            df_years = pd.to_numeric(df['year'], errors='coerce').fillna(0)
            era_df_mask = pd.Series([True] * len(df), index=df.index)
            if yr_min: era_df_mask &= df_years >= yr_min
            if yr_max: era_df_mask &= df_years <= yr_max

            # Language filter on full df too
            if query_lang and 'original_language' in df.columns:
                ql = str(query_lang).lower().strip()
                ql_group = _LANG_GROUPS.get(ql, 'other')
                df_lang = df['original_language'].astype(str).str.lower().str.strip()
                # Try exact language first, then group
                exact_mask = era_df_mask & (df_lang == ql)
                if exact_mask.sum() >= top_n * 3:
                    era_df_mask = exact_mask
                else:
                    def _gl(l): return _LANG_GROUPS.get(l, 'other') == ql_group
                    group_mask = era_df_mask & df_lang.apply(_gl)
                    era_df_mask = group_mask if group_mask.sum() >= top_n else exact_mask

            era_pool = df[era_df_mask].copy()
            era_pool['_sim'] = 0.3   # baseline sim for injected films

            # Merge: keep FAISS in-era films (higher sim) + inject era pool
            in_era_candidates = candidates[era_mask].copy() if era_count > 0 else pd.DataFrame()
            already_titles = set(in_era_candidates['title_clean'].tolist()) if not in_era_candidates.empty else set()
            inject = era_pool[~era_pool['title_clean'].isin(already_titles)]

            # Sort injected by rating+popularity as proxy for quality
            if not inject.empty:
                va_i = pd.to_numeric(inject['vote_average'], errors='coerce').fillna(0)
                vc_i = pd.to_numeric(inject['vote_count'],   errors='coerce').fillna(0)
                inject = inject.copy()
                inject['_inject_score'] = va_i * 0.7 + (vc_i / 10000).clip(0, 3) * 0.3
                inject = inject.sort_values('_inject_score', ascending=False).head(top_n * 8)
                inject = inject.drop(columns=['_inject_score'], errors='ignore')

            candidates = pd.concat([in_era_candidates, inject], ignore_index=True)
            print('[pool] Era inject: {} in-era films (FAISS={} + injected={})'.format(
                len(candidates), len(in_era_candidates), len(inject)))

    # ── Quality filter ────────────────────────────────────────────────────────
    va       = pd.to_numeric(candidates['vote_average'], errors='coerce').fillna(0)
    vc       = pd.to_numeric(candidates['vote_count'],   errors='coerce').fillna(0)
    min_va   = 5.5
    min_vc   = 100
    quality  = candidates[(va >= min_va) & (vc >= min_vc)]
    if len(quality) >= top_n:
        candidates = quality

    # ── Step 4: weighted scoring ──────────────────────────────────────────────
    scores = []
    for _, row in candidates.iterrows():
        rd         = row.to_dict()
        sim        = float(rd.get('_sim', 0))
        vibe       = _vibe_score(query_row, rd)
        rating     = float(pd.to_numeric(rd.get('vote_average', 6), errors='coerce') or 6)
        vote_count = float(pd.to_numeric(rd.get('vote_count', 0),   errors='coerce') or 0)
        genre_ovlp = _genre_overlap_score(rd, query_genres)
        era_s      = _era_score(rd.get('year', 0), era)
        lang_s       = _lang_affinity_score(query_lang, rd.get('original_language', 'en'))
        native_boost = (str(query_lang or 'en').lower().strip() != 'en')
        score        = _weighted_score(sim, vibe, rating, vote_count, genre_ovlp, era_s, lang_s,
                                       native_boost=native_boost)
        rd['_score'] = score
        scores.append(rd)

    scores.sort(key=lambda x: x['_score'], reverse=True)
    top_pool = scores[:top_n * 3]

    print('[pool] {} candidates scored. Top 5: {}'.format(
        len(top_pool), [r.get('title') for r in top_pool[:5]]))

    return pd.DataFrame(top_pool)


# Keep tfidf_fallback as legacy alias for any external callers
def tfidf_fallback(query_idx, query_clean, exclude_titles, era,
                   genres_exclude, genres_require, top_n, query_genres=None, query_lang='en'):
    qr = safe_row(query_idx)
    return build_candidate_pool(
        query_idx      = query_idx,
        query_clean    = query_clean,
        query_row      = qr,
        exclude_titles = exclude_titles,
        era            = era,
        genres_exclude = genres_exclude,
        genres_require = genres_require,
        top_n          = top_n,
        query_genres   = query_genres,
        query_lang     = query_lang,
    )


# ── Groq Candidate Ranker ──────────────────────────────────────────────────────
# TF-IDF generates candidate pool → Groq ranks by emotional similarity.
# Groq CANNOT hallucinate: it can only rank from the provided list.

_RANK_PROMPT = """You are a cinematic vibe-matching engine.
Your goal is to recommend films that match the EMOTIONAL VIBE of the reference film,
not superficial similarities like country, language, setting, or era.

------------------------------------------------------------
STRICT RULES
------------------------------------------------------------
1. Only choose movies from the provided candidate list.
2. Return EXACTLY {n} movies.
3. Do NOT hallucinate titles.
4. Return JSON only.
5. If few perfect matches exist, choose the closest emotional matches.
6. Era is a STRONG preference. See ERA HANDLING for exact rules.
7. Genre compatibility is a HARD requirement. See GENRE RULES below.
8. Never return fewer than {n} results.

------------------------------------------------------------
ERA HANDLING — STRICT RULES
------------------------------------------------------------
SELECTED ERA tells you which years the user wants. Follow this exactly:

- "Any Era"  -> No restriction. Rank by vibe only.
- "Classic"  -> STRONGLY prefer films before 1990.
               Only use post-1990 films if fewer than {n} good matches exist pre-1990.
- "90s"      -> STRONGLY prefer 1990-1999.
               Only use films outside 1990-1999 if fewer than {n} good matches exist.
- "2000s"    -> STRONGLY prefer 2000-2009. Same fallback rule.
- "2010s"    -> STRONGLY prefer 2010-2019. Same fallback rule.
- "Modern"   -> STRONGLY prefer 2018-2021. Same fallback rule.
- "New"      -> STRONGLY prefer 2020 onward.
               DO NOT rank films from before 2020 in the top 3 unless NO good matches exist.

ERA ENFORCEMENT EXAMPLE:
  If era=New and you see "Jab We Met (2007)" and "Tu Jhoothi Main Makkaar (2023)"
  with similar vibe — Tu Jhoothi MUST rank higher because it matches the era.
  "Jab We Met" should only appear if fewer than {n} films from 2020+ exist.

IMPORTANT: The year is shown for every candidate. Read it. Use it.

------------------------------------------------------------
CRITICAL MATCHING PRINCIPLE
------------------------------------------------------------
Match the TONE and EMOTIONAL ATMOSPHERE of the reference film.

Tone vocabulary:
  whimsical | dreamy | melancholic | romantic | playful | nostalgic
  dark | intense | bittersweet | warm | chaotic | surreal | hopeful
  tender | gritty | haunting | lighthearted | epic | intimate

DO NOT match based on:
- country or city of setting
- shared language
- shared cast or director
- same historical period
- surface plot similarity
- same genre label alone

------------------------------------------------------------
GENRE RULES — HARD FILTER
------------------------------------------------------------
The reference film has genres. Do NOT recommend films with incompatible genre DNA.

Genre compatibility matrix:
  Romance/RomCom       → OK: Comedy, Drama, Family | AVOID: Horror, Thriller, War, Crime
  Comedy               → OK: Romance, Drama, Family, Adventure | AVOID: Horror, War
  Inspirational Drama  → OK: History, Biography, Sport, Achievement | AVOID: Romance-focused, Horror, RomCom
  Social Drama         → OK: Family, Comedy, History | AVOID: Action, Horror, Thriller
  Action/Thriller      → OK: Crime, Mystery, Adventure | AVOID: Romance-focused films
  Horror               → OK: Thriller, Mystery, Fantasy | AVOID: RomCom, Family Comedy
  Family               → OK: Comedy, Drama, Adventure, Animation | AVOID: Horror, Thriller

  CRITICAL — "Drama" is NOT one genre. Identify the DRAMA SUB-TYPE:
    Inspirational/Achievement Drama: films about teams, missions, scientists, athletes overcoming odds
    Social Drama:                    films about class, family, society, relationships
    Romantic Drama:                  love story as primary engine
    Historical Drama:                period setting, real events
  These sub-types are NOT interchangeable. A social drama is WRONG for an inspirational drama query.

GENRE EXAMPLES:
  Reference: Yeh Jawaani Hai Deewani (Romance, Comedy, Drama)
    CORRECT: Tu Jhoothi Main Makkaar (Romance, Comedy) ✓ — genre match
    WRONG:   A crime thriller with minor romantic subplot ✗ — wrong genre DNA
    WRONG:   A war film with a love story ✗ — wrong genre DNA

  Reference: Mission Mangal (Drama, History) — INSPIRATIONAL ACHIEVEMENT DRAMA
    Tone: triumphant, team-driven, against-the-odds, pride, scientific mission
    CORRECT: Chak De India, Dangal, Rocket Singh, October Sky, Hidden Figures ✓
    WRONG:   Manto (partition drama — same language, wrong sub-type) ✗
    WRONG:   Veer-Zaara (romantic drama — same language, wrong sub-type) ✗
    WRONG:   Hindi Medium (social comedy-drama — same language, wrong sub-type) ✗
    RULE: "Drama" shared between Mission Mangal and Veer-Zaara means NOTHING.
          The inspirational achievement sub-type must match.

  Reference: Parasite (Drama, Thriller)
    CORRECT: Burning (Drama, Mystery, Thriller) ✓ — genre match
    WRONG:   A romantic comedy with dark undertones ✗ — wrong genre DNA



RULE: If a candidate's PRIMARY genre conflicts with the reference's PRIMARY genre,
      do NOT rank it in top 3 regardless of vibe score.
      For Drama films: always identify the SUB-TYPE and match it.

------------------------------------------------------------
EMOTIONAL MATCHING PRIORITY (highest to lowest)
------------------------------------------------------------
1. Genre compatibility               (HARD REQUIREMENT — see GENRE RULES)
2. Tone and emotional atmosphere     (MOST IMPORTANT after genre check)
3. Relationship dynamic              (romantic pair, friends, family, ensemble, self)
4. Emotional warmth vs darkness      balance
5. Story style                       (quirky, surreal, grounded, epic, intimate)
6. Narrative arc                     (self-discovery, reunion, tragedy, healing, coming-of-age)
7. Era preference                    (see ERA HANDLING — strict)

------------------------------------------------------------
TONE IDENTIFICATION — STEP BY STEP
------------------------------------------------------------
Step 1: Read the reference film title, overview, and genres.
Step 2: Identify its DOMINANT TONE from the tone vocabulary above.
Step 3: Identify its SECONDARY TONE if present.
Step 4: Rank only candidates whose tone matches Step 2 and Step 3.
Step 5: Apply era as a soft tiebreaker at the end.

------------------------------------------------------------
TONE GUARDRAILS WITH EXAMPLES
------------------------------------------------------------

--- WHIMSICAL / MAGICAL / ROMANTIC FILMS ---

Reference: Amelie (2001)
Dominant tone: whimsical, dreamy, romantic, visually playful
CORRECT matches: Big Fish, Midnight in Paris, The Grand Budapest Hotel,
                 Chocolat, Beasts of the Southern Wild, Ruby Sparks
WRONG matches:   A Paris-based political drama (same city, wrong tone)
                 A French war film (same language, wrong tone)
                 A biographical struggle film (both "artistic", wrong tone)
Rule: Do NOT match just because a film is also set in Paris, also French,
      or also considered "artistic". Match the whimsy and warmth.

Reference: Yeh Jawaani Hai Deewani (2013)
Dominant tone: warm, playful, nostalgic, romantic, youthful energy
CORRECT tone profile: warm romantic comedies, youthful ensemble films,
                      travel-and-friendship adventures, breezy coming-of-age stories
WRONG tone profile:   sports biopics (same language, wrong tone)
                      spy thrillers (same language, wrong tone)
                      heavy social realism (same language, wrong tone)
                      dark crime sagas (same language, wrong tone)
Rule: Sharing a language, era, or country is NOT enough.
      The warm playful nostalgic tone must match.
NOTE: Do NOT suggest specific titles from memory. Only rank from the candidate list.

Reference: Midnight in Paris (2011)
Dominant tone: dreamy, romantic, nostalgic, magical realism, witty
CORRECT matches: Amelie, The Grand Budapest Hotel, Roman Holiday,
                 Before Sunset, Vicky Cristina Barcelona
WRONG matches:   A gritty Paris crime thriller (same city, wrong tone)
                 A World War 2 drama set in France (same country, wrong tone)
Rule: Setting is irrelevant. Dreaminess and romantic nostalgia must match.

--- DARK / INTENSE / PSYCHOLOGICAL FILMS ---

Reference: Parasite (2019)
Dominant tone: dark, tense, class-charged, socially sharp, suffocating
CORRECT matches: Burning, Memories of Murder, A Tale of Two Sisters,
                 Snowpiercer, The Handmaiden, Shoplifters
WRONG matches:   A lighthearted Korean romantic comedy (same country, wrong tone)
                 A Korean family drama with warm resolution (same country, wrong tone)
Rule: Being Korean is NOT enough. The darkness and class tension must match.

Reference: Requiem for a Dream (2000)
Dominant tone: harrowing, dark, spiraling, tragic, visceral, intense
CORRECT matches: Black Swan, Trainspotting, Darren Aronofsky films,
                 Irreversible, Enter the Void, Christiane F
WRONG matches:   A hopeful addiction recovery story (similar topic, opposite tone)
                 An uplifting sports film about overcoming struggle (wrong tone)
Rule: Similar topic (addiction, struggle) is NOT enough. The spiraling darkness must match.

Reference: Oldboy (2003)
Dominant tone: dark, twisted, psychologically intense, haunting, disturbing
CORRECT matches: Memories of Murder, I Saw the Devil, A Bittersweet Life,
                 Se7en, Gone Girl, Black Swan
WRONG matches:   A stylish Korean action comedy (same country, wrong tone)
                 A Korean romance (same language, wrong tone)
Rule: Do NOT match just because a film is Korean, stylish, or "intense cinema".

--- YOUTH / COMING-OF-AGE / FRIENDSHIP FILMS ---

Reference: The Breakfast Club (1985)
Dominant tone: rebellious, warm, nostalgic, honest, tender, youthful
CORRECT matches: Ferris Bueller's Day Off, Say Anything, Sixteen Candles,
                 Stand by Me, Dazed and Confused, Lady Bird
WRONG matches:   A dark teen drama about trauma and abuse (teens present, wrong tone)
                 A military coming-of-age film (coming-of-age label, wrong tone)
Rule: Coming-of-age label is NOT enough. The warmth and youthful rebellion must match.

Reference: Dil Chahta Hai (2001)
Dominant tone: carefree, playful, friendship-driven, nostalgic, breezy
CORRECT tone profile: carefree friendship dramas, breezy ensemble comedies,
                      nostalgic coming-of-age stories, lighthearted road trips
WRONG tone profile:   epic historical struggles (same language, wrong tone)
                      tragic romances (same era, wrong tone)
                      heavy dramas (same language, wrong tone)
Rule: Being a Hindi film from the same era is NOT enough.
      The carefree friendship energy must match.
NOTE: Do NOT suggest specific titles from memory. Only rank from the candidate list.

--- BITTERSWEET / MELANCHOLIC / HEALING FILMS ---

Reference: Eternal Sunshine of the Spotless Mind (2004)
Dominant tone: bittersweet, melancholic, romantic, surreal, emotionally aching
CORRECT matches: Her, Lost in Translation, Blue Valentine,
                 500 Days of Summer, Vanilla Sky, The Science of Sleep
WRONG matches:   A feel-good romantic comedy (romantic label, wrong tone)
                 A dark psychological thriller (surreal label, wrong tone)
Rule: Romantic label alone is NOT enough. The bittersweet ache and surreal quality must match.

Reference: Lost in Translation (2003)
Dominant tone: melancholic, quiet, tender, lonely, intimate, understated
CORRECT matches: Her, Eternal Sunshine, Paterson, Before Sunrise,
                 The Virgin Suicides, In the Mood for Love
WRONG matches:   A loud action film set in Tokyo (same city, wrong tone)
                 A fast-paced Japanese thriller (same country, wrong tone)
Rule: Being set in Japan or Tokyo is NOT enough. The quiet melancholy must match.

--- EPIC / ADVENTURE / ASPIRATIONAL FILMS ---

Reference: Zindagi Na Milegi Dobara (2011)
Dominant tone: aspirational, freeing, joyful, warm, friendship-driven, travel-energy
CORRECT tone profile: joyful travel-and-friendship adventures, liberating self-discovery,
                      warm ensemble comedies, aspirational life-affirming stories
WRONG tone profile:   survival thrillers (travel present, wrong tone)
                      dark war films (adventure present, wrong tone)
Rule: Travel or adventure setting is NOT enough. The joyful liberating energy must match.
NOTE: Do NOT suggest specific titles from memory. Only rank from the candidate list.

--- INSPIRATIONAL / ACHIEVEMENT / MISSION FILMS ---

Reference: Mission Mangal (2019)
Dominant tone: triumphant, team-driven, against-the-odds, nationalistic pride, hopeful
CORRECT tone profile: scientist/athlete/team achievement films, mission-driven narratives,
                      underdog triumph, institutional pride, ensemble effort
WRONG tone profile:   romantic dramas (Drama label shared, wrong sub-type)
                      social comedies (Hindi film, wrong tone entirely)
                      partition/historical dramas (History label shared, wrong tone)
                      quiet introspective films (same era, wrong energy)
Rule: The TRIUMPHANT MISSION ENERGY must match. Not just "Drama" or "Hindi" or "History".
NOTE: Do NOT suggest specific titles from memory. Only rank from the candidate list.

Reference: Dangal (2016)
Dominant tone: inspirational, gritty determination, family pride, underdog, sports triumph
CORRECT tone profile: sports biopics, against-the-odds achievement, father-daughter bond,
                      real-life triumph narratives
WRONG tone profile:   soft family dramas (family present, wrong energy)
                      romantic films (same era, wrong tone)
NOTE: Do NOT suggest specific titles from memory. Only rank from the candidate list.

--- TRAGIC / SACRIFICIAL / HEAVY FILMS ---

Reference: Devdas (2002)
Dominant tone: tragic, operatic, heavy, self-destructive, melancholic, doomed romance
CORRECT matches: Mughal-E-Azam, Umrao Jaan, Pakeezah,
                 Romeo and Juliet, Layla and Majnun adaptations
WRONG matches:   A modern Hindi romantic comedy (romantic label, wrong tone)
                 A Bollywood dance film (same industry, wrong tone)
Rule: Being a Bollywood romance is NOT enough. The tragic operatic weight must match.

------------------------------------------------------------
LANGUAGE HANDLING — STRICT RULES
------------------------------------------------------------
The REFERENCE MOVIE has a language. Your job is to reflect that language back.

Language groups:
  South Asian : Hindi, Tamil, Telugu, Malayalam, Kannada, Bengali, Marathi
  Western     : English, French, Spanish, Italian, German, Portuguese
  East Asian  : Chinese, Japanese, Korean

PRIORITY ORDER — fill your final list in this order:
  1. Same language as reference (e.g. Hindi→Hindi) — fill as many slots as possible
  2. English — always acceptable as secondary option
  3. Same language group (e.g. Hindi→Tamil) — only if slots 1+2 still not filled
  4. Different group — ONLY if absolutely no other options exist

HARD RULES:
- If reference is Hindi: at LEAST 60% of results must be Hindi or English.
- If reference is Korean: at LEAST 60% must be Korean or English.
- NEVER fill majority of results with a language from a DIFFERENT group.
- A French/German/Thai film must NEVER appear in top 3 for a Hindi reference film.

EXAMPLE — Reference: Hindi (Taare Zameen Par)
  CORRECT: Taare Zameen Par → Udaan (Hindi), Nil Battey Sannata (Hindi), Stanley Ka Dabba (Hindi), Soul (English), Dangal (Hindi)
  WRONG:   Taare Zameen Par → The Holdovers (English), Marcel the Shell (English), Bridge to Terabithia (English) — too many English, no Hindi

EXAMPLE — Reference: Japanese (Your Name)
  CORRECT: Your Name → A Silent Voice (Japanese), 5cm/s (Japanese), Her (English)
  WRONG:   Your Name → Dilwale (Hindi), Taare Zameen Par (Hindi) — completely wrong group

------------------------------------------------------------
FINAL REMINDER
------------------------------------------------------------
The single most common mistake is matching films by:
- Same country / language / city
- Same genre label
- Same era
- Superficially similar plot

ALWAYS ask: Does the EMOTIONAL TONE match?
If the answer is no — do not rank it highly, regardless of everything else.

------------------------------------------------------------
INPUT DATA
------------------------------------------------------------
REFERENCE MOVIE:
Title: {ref_title}
Year: {ref_year}
Language: {ref_lang}
Genres: {ref_genres}   ← USE THIS for genre compatibility check (GENRE RULES above)
Overview: {ref_overview}

SELECTED ERA:
{selected_era}

CANDIDATE MOVIES:
{candidates}

------------------------------------------------------------
TASK
------------------------------------------------------------
Step 1: Identify the PRIMARY and SECONDARY genres of the reference film.
Step 2: Identify the dominant emotional tone (from tone vocabulary).
Step 3: Filter candidates — remove any whose primary genre conflicts with reference (GENRE RULES).
Step 4: Among remaining candidates, keep only those matching the era (ERA HANDLING).
Step 5: Rank by emotional tone match + genre fit.
Step 6: Only if fewer than {n} candidates survive steps 3-4, relax era restriction first, then genre.

Return ONLY this JSON:
{{
  "recommendations": [
    {{"title": "Movie Name", "year": 2011, "rating": 7.8, "rank": 1}}
  ]
}}
Return EXACTLY {n} items. No commentary. No markdown. JSON only."""


def _derive_ranking_axes(fp):
    """Map 6-axis fingerprint to 5 ranking axes: Intensity, Romance, Darkness, Humor, Hope."""
    if not fp:
        return dict(intensity=5, romance=5, darkness=5, humor=3, hope=5)
    intensity = int(fp.get('emotional_intensity', 5))
    darkness  = int(fp.get('darkness', 5))
    warmth    = int(fp.get('emotional_warmth', 5))
    pc        = int(fp.get('psychological_complexity', 5))
    tone      = str(fp.get('tone_direction', 'mixed')).lower()
    romance   = round(min(10, warmth * 0.7 + (10 - pc) * 0.3))
    humor     = round(min(10, max(0, (10 - darkness) * 0.5 + (10 - intensity) * 0.3)))
    if 'optimistic' in tone:
        hope = round(min(10, 8 + (10 - darkness) * 0.2))
    elif 'dark' in tone:
        hope = round(max(0, 3 - darkness * 0.2))
    else:
        hope = round(min(10, max(0, (10 - darkness) * 0.6 + warmth * 0.2)))
    return dict(intensity=intensity, romance=int(romance), darkness=darkness,
                humor=int(humor), hope=int(hope))


def groq_rank_candidates(query_title, query_row, candidate_rows, top_n=5, era=None):
    """
    Groq ranks a pre-built TF-IDF candidate pool by emotional and narrative similarity.
    Cannot hallucinate - can only pick from the provided numbered list.
    Returns ordered list of row dicts.
    """
    if not GROQ_API_KEY or not candidate_rows:
        return candidate_rows[:top_n]

    overview = str(query_row.get('overview', '') or '')[:200]
    fp       = get_emotional_fingerprint(query_row.get('title', query_title), overview)
    axes     = _derive_ranking_axes(fp)
    genres   = ', '.join(_get_genres(query_row)) or 'Unknown'
    ref_year = int(query_row.get('year', 0) or 0)

    # Era label — must match the exact keywords in ERA HANDLING section of the prompt
    era_label = {
        'new':     'New',
        'modern':  'Modern',
        '2010s':   '2010s',
        '2000s':   '2000s',
        '90s':     '90s',
        'classic': 'Classic',
    }.get((era or '').lower(), 'Any Era')

    # Candidate list: numbered, with rating and 150-char overview
    lines = []
    for i, row in enumerate(candidate_rows, 1):
        ov     = str(row.get('overview', '') or '')[:90].replace('\n', ' ')
        yr     = int(row.get('year', 0) or 0)
        rating = float(row.get('vote_average', 0) or 0)
        lang_code = str(row.get('original_language', '') or '').strip()
        lang_display = {
            'hi': 'Hindi', 'ta': 'Tamil', 'te': 'Telugu', 'ml': 'Malayalam',
            'kn': 'Kannada', 'bn': 'Bengali', 'mr': 'Marathi',
            'en': 'English', 'fr': 'French', 'es': 'Spanish', 'it': 'Italian',
            'de': 'German', 'pt': 'Portuguese',
            'zh': 'Chinese', 'ja': 'Japanese', 'ko': 'Korean',
        }.get(lang_code, lang_code.upper() if lang_code else '?')
        # Include genres so Groq can apply GENRE RULES
        genres_raw = row.get('genres', '') or ''
        if isinstance(genres_raw, list):
            genres_str = ', '.join(str(g) for g in genres_raw[:3])
        else:
            genres_str = str(genres_raw).replace('|', ',').strip()[:40]
        lines.append('{}. {} ({}) | Lang: {} | Genres: {} | Rating: {:.1f} | {}'.format(
            i, row.get('title', ''), yr, lang_display, genres_str, rating, ov))

    ref_lang_code = str(query_row.get('original_language') or 'en').strip() or 'en'
    ref_lang_label = {
        'hi': 'Hindi', 'ta': 'Tamil', 'te': 'Telugu', 'ml': 'Malayalam',
        'kn': 'Kannada', 'bn': 'Bengali', 'mr': 'Marathi',
        'en': 'English', 'fr': 'French', 'es': 'Spanish', 'it': 'Italian',
        'de': 'German', 'pt': 'Portuguese',
        'zh': 'Chinese', 'ja': 'Japanese', 'ko': 'Korean',
    }.get(ref_lang_code, ref_lang_code.upper())
    prompt = _RANK_PROMPT.format(
        ref_title    = str(query_row.get('title', query_title)).replace('{','(').replace('}',')'),
        ref_year     = ref_year,
        ref_lang     = ref_lang_label,
        ref_overview = str(query_row.get('overview','') or '')[:200].replace('{','(').replace('}',')'),
        ref_genres   = genres,
        selected_era = era_label,
        candidates   = '\n'.join(lines),
        n            = top_n,
    )
    print('[ranker] era={} axes: i={intensity} r={romance} d={darkness} h={humor} hope={hope}'.format(
        era_label, **axes))

    raw = _groq(
        [{'role': 'system', 'content': 'Return only valid JSON. No markdown. No text outside JSON.'},
         {'role': 'user',   'content': prompt}],
        temperature=0.1, max_tokens=500,
        model='llama-3.3-70b-versatile',  # 70b for nuanced tone matching
    )
    result = _parse_json_obj(raw)
    if not result or not isinstance(result.get('recommendations'), list):
        print('[ranker] Parse failed — using TF-IDF order')
        return candidate_rows[:top_n]

    # Title lookup (normalized)
    title_to_row = {}
    for row in candidate_rows:
        key = re.sub(r'[^a-z0-9 ]', '', str(row.get('title', '')).lower().strip())
        title_to_row[key] = row

    # Sort by rank field
    recs = result['recommendations']
    try:
        recs = sorted(recs, key=lambda x: int(x.get('rank', 999)))
    except Exception:
        pass

    ranked      = []
    ranked_seen = set()
    for rec in recs:
        t   = str(rec.get('title', '')).strip()
        key = re.sub(r'[^a-z0-9 ]', '', t.lower().strip())
        if key in ranked_seen:
            print('[ranker] DUP "{}" — skipped'.format(t))
            continue
        if key in title_to_row:
            ranked.append(title_to_row[key])
            ranked_seen.add(key)
            print('[ranker] #{} "{}" rating={}'.format(
                rec.get('rank', len(ranked)), t, rec.get('rating', '?')))
        else:
            # Resolve from DB — catches ZNMD, Dil Dhadakne Do etc that TF-IDF misses
            try:
                idx = resolve(t, save_on_inject=False)
                if idx is not None:
                    rd = safe_row(idx)
                    if rd and rd.get('title'):
                        title_to_row[key] = rd
                        ranked.append(rd)
                        ranked_seen.add(key)
                        print('[ranker] #{} "{}" (DB resolved) rating={}'.format(
                            rec.get('rank', len(ranked)), t, rd.get('vote_average', '?')))
                    else:
                        print('[ranker] SKIP "{}" — not in pool'.format(t))
                else:
                    print('[ranker] SKIP "{}" — not in pool'.format(t))
            except Exception:
                print('[ranker] SKIP "{}" — not in pool'.format(t))

    if not ranked:
        print('[ranker] No matches — using TF-IDF order')
        return candidate_rows[:top_n]

    # Fill remaining slots from TF-IDF order if fewer than top_n
    ranked_keys = {re.sub(r'[^a-z0-9 ]', '', str(r.get('title','')).lower().strip()) for r in ranked}
    for row in candidate_rows:
        if len(ranked) >= top_n:
            break
        key = re.sub(r'[^a-z0-9 ]', '', str(row.get('title','')).lower().strip())
        if key not in ranked_keys:
            ranked.append(row)

    print('[ranker] Final: {}'.format([r.get('title') for r in ranked[:top_n]]))
    return ranked[:top_n]


# ── compound recent suggestions ────────────────────────────────────────────────

def compound_suggest_with_meta(prompt, lang, n=10):
    """Get recent film title suggestions from Groq. Metadata comes from DB lookup later.
    
    Removed the second enrichment Groq call (max_tokens=2000) — it was the single
    biggest contributor to timeouts. Title+year is enough for DB resolution.
    """
    raw = _groq([{'role': 'user', 'content': prompt + '\nReturn ONLY JSON: [{"title":"","year":0}].'}],
                temperature=0.3, max_tokens=600)
    title_list = _parse_json_arr(raw)
    title_list = [f for f in title_list if isinstance(f.get('year'), int) and f['year'] >= 2018]
    print('[compound] {} titles: {}'.format(len(title_list), [f.get('title') for f in title_list[:5]]))
    if not title_list:
        return []

    # Return title+year only — DB resolution fills metadata downstream
    return [{'title': f.get('title', ''), 'year': f.get('year', 0),
             'genres': [], 'overview': '', 'original_language': lang}
            for f in title_list]


# ── Main recommend function ────────────────────────────────────────────────────


# ── Metadata Enrichment ────────────────────────────────────────────────────────
# Called when a film's key fields are missing — happens for films with sparse
# parquet data (e.g. original_language=None, empty genres, no overview).
# Strategy: TMDB first (accurate), Groq as fallback (best-effort).

_SPARSE_FIELDS = ('original_language', 'genres', 'overview', 'director')
_ENRICH_CACHE  = {}   # title_clean → enriched dict, avoid re-fetching same film


def _safe_val(v):
    """Convert any value (including numpy scalar/array) to plain Python — None if empty."""
    import numpy as _np
    if v is None:
        return None
    if isinstance(v, _np.ndarray):
        return v.tolist() if v.size > 0 else None
    if isinstance(v, _np.generic):
        return v.item()
    return v


def _is_sparse(row: dict) -> bool:
    """Return True if any critical field is missing/empty. Numpy-safe."""
    lang  = _safe_val(row.get('original_language'))
    genre = _safe_val(row.get('genre_list')) or _safe_val(row.get('genres'))
    ov    = _safe_val(row.get('overview'))
    lang_empty  = (not lang  or str(lang).strip().lower()  in ('', 'nan', 'none'))
    genre_empty = (not genre or (isinstance(genre, list) and len(genre) == 0)
                             or str(genre).strip() in ('', 'nan', 'none', '[]'))
    ov_empty    = (not ov    or str(ov).strip().lower()    in ('', 'nan', 'none'))
    return lang_empty or genre_empty or ov_empty


def enrich_if_sparse(query_row: dict, title: str, year=None) -> dict:
    """
    If query_row is missing critical metadata, fetch from TMDB → Groq fallback.
    Returns an enriched copy of query_row (original df not modified here).
    Patches: original_language, genres, genre_list, overview, director,
             tagline, keywords, cast, poster_url.
    """
    if not _is_sparse(query_row):
        return query_row   # already complete — skip

    cache_key = re.sub(r'[^a-z0-9]', '', str(title).lower())
    if cache_key in _ENRICH_CACHE:
        print('[enrich] Cache hit for "{}"'.format(title))
        return {**query_row, **_ENRICH_CACHE[cache_key]}

    print('[enrich] Sparse metadata for "{}". Fetching from TMDB...'.format(title))
    tmdb = tmdb_search(title, year)

    if 'error' in tmdb:
        print('[enrich] TMDB failed: {} — trying Groq...'.format(tmdb['error']))
        enriched = groq_enrich(title, {})
    else:
        print('[enrich] TMDB found: lang={} genres={}'.format(
            tmdb.get('original_language'), tmdb.get('genres')))
        # If TMDB gave us what we need, use it directly
        enriched = tmdb

    if not enriched or 'error' in enriched:
        print('[enrich] Could not enrich "{}". Using sparse data.'.format(title))
        return query_row

    # Build patch — only overwrite fields that were missing in query_row
    patch = {}
    field_map = {
        'original_language': 'original_language',
        'overview':          'overview',
        'director':          'director',
        'tagline':           'tagline',
        'poster_url':        'poster_url',
    }
    for src_key, dst_key in field_map.items():
        val = enriched.get(src_key)
        existing = query_row.get(dst_key)
        if val and (not existing or str(existing).strip().lower() in ('', 'nan', 'none')):
            patch[dst_key] = val

    # Genres — patch both 'genres' string and 'genre_list'
    if enriched.get('genres'):
        existing_genres = _safe_val(query_row.get('genre_list')) or _safe_val(query_row.get('genres'))
        is_empty = (not existing_genres or
                    (isinstance(existing_genres, list) and len(existing_genres) == 0) or
                    str(existing_genres).strip() in ('', 'nan', 'none', '[]'))
        if is_empty:
            genre_list = enriched['genres'] if isinstance(enriched['genres'], list)                          else [g.strip() for g in str(enriched['genres']).split(',') if g.strip()]
            patch['genre_list'] = genre_list
            patch['genres']     = ', '.join(genre_list)

    if patch:
        print('[enrich] Patched fields: {}'.format(list(patch.keys())))
        _ENRICH_CACHE[cache_key] = patch
        return {**query_row, **patch}

    return query_row


def recommend_by_vibe(query_title, top_n=5, exclude_titles=None,
                      year=None, era=None, refine=None):
    global LAST_SESSION
    _size_before   = len(df)
    genres_exclude = []
    genres_require = []

    # ── Refine ────────────────────────────────────────────────────────────────
    if refine:
        if not LAST_SESSION.get('query'):
            return None, "No previous session to refine. Please search for a film first."
        parsed         = groq_parse_refinement(refine, LAST_SESSION)
        action         = parsed.get('action', 'more')
        if action == 'new_mood':
            from .mood import mood_based_recommend
            return mood_based_recommend(parsed.get('new_mood') or refine, top_n=top_n)
        query_title    = LAST_SESSION['query']
        exclude_titles = list(LAST_SESSION.get('shown', [])) + (exclude_titles or [])
        genres_exclude = parsed.get('genres_exclude') or []
        genres_require = parsed.get('genres_require') or []
        top_n          = LAST_SESSION.get('top_n', top_n)
        # Restore era and year from session — critical for Show More on ambiguous titles
        era            = era or LAST_SESSION.get('era')
        year           = year or LAST_SESSION.get('year')
        clean          = re.sub(r'[^a-z0-9 ]', '', query_title.lower().strip())
        _VIBE_CACHE.pop('v12_{}_{}_{}'.format(clean, top_n, era or ''), None)
        print('[vibe] Refine: era={} exclude={}'.format(era, exclude_titles[:3]))

    # ── Disambiguation ────────────────────────────────────────────────────────
    _, disambig = check_disambiguation(query_title)
    if disambig is not None and year is None:
        sort_col = 'vote_count' if 'vote_count' in disambig.columns else 'vote_average'
        best     = disambig.sort_values(sort_col, ascending=False).iloc[0]

        # If there are genuinely different films (different years or languages), ask user
        years_in_db = disambig['year'].dropna().astype(int).tolist()
        langs_in_db = disambig.get('original_language', pd.Series()).dropna().tolist() if 'original_language' in disambig.columns else []
        has_ambiguity = len(set(years_in_db)) > 1 or (len(set(langs_in_db)) > 1 if langs_in_db else False)

        if has_ambiguity:
            options = disambig[['title', 'year', 'vote_average']].to_dict('records')
            return None, 'DISAMBIG: Multiple films named "{}". Which did you mean?'.format(query_title), options

        # Only one meaningfully distinct film — auto-select best
        year = int(best['year']) if not pd.isna(best['year']) else None
        print('[vibe] Disambig auto-select -> {} ({})'.format(best['title'], year))

    # ── Resolve query film ─────────────────────────────────────────────────────
    query_clean = _normalize_title(query_title)
    cache_key   = 'v12_{}_{}_{}_{}'.format(query_clean, top_n, era or '', year or '')

    # Cache check — runs regardless of whether year was provided
    if cache_key in _VIBE_CACHE and not refine and not exclude_titles:
        print('[cache] Hit: {}'.format(cache_key))
        return _VIBE_CACHE[cache_key]

    if year:
        # Try normalized clean first, then raw clean for accented titles
        query_clean_raw = re.sub(r'[^a-z0-9 ]', '', query_title.lower().strip())
        mask  = df['title_clean'].isin([query_clean, query_clean_raw])
        mask &= (df['year'] >= year - 1) & (df['year'] <= year + 1)
        matches = df[mask]
        if not matches.empty:
            query_idx = int(matches['vote_count'].idxmax() if 'vote_count' in matches.columns else matches.index[0])
        else:
            # Exact match failed — try partial title + year (handles "Agnipath", "Batman" etc.)
            partial_mask  = df['title_clean'].str.contains(query_clean_raw, na=False)
            partial_mask &= (df['year'] >= year - 1) & (df['year'] <= year + 1)
            partial = df[partial_mask]
            if not partial.empty:
                query_idx = int(partial['vote_count'].idxmax() if 'vote_count' in partial.columns else partial.index[0])
                print('[vibe] Partial+year match: {} ({})'.format(df.loc[query_idx, 'title'], df.loc[query_idx, 'year']))
            else:
                query_idx = resolve(query_title)
    else:
        query_idx = resolve(query_title)

    if query_idx is None:
        # Film not in local DB — try fetching from TMDB and injecting it
        print('[vibe] Not in DB, trying TMDB for: {}'.format(query_title))
        tmdb_data = tmdb_search(query_title)
        if 'error' not in tmdb_data:
            query_idx = inject_movie(tmdb_data, save=True)
            if query_idx is not None:
                print('[vibe] Injected query film from TMDB: {}'.format(tmdb_data.get('title')))

    if query_idx is None:
        persist_if_new(_size_before)
        return None, "Could not find '{}'. Try adding the release year.".format(query_title)

    query_row    = safe_row(query_idx)

    # ── Enrich sparse metadata before pipeline runs ───────────────────────────
    query_row = enrich_if_sparse(
        query_row = query_row,
        title     = query_row.get('title', query_title),
        year      = int(query_row.get('year', 0) or 0) or year or None,
    )

    query_genres = _query_primary_genres(query_row)
    query_lang   = str(query_row.get('original_language') or 'en').strip() or 'en'
    print('[vibe] Query: "{}" idx={} lang={} genres={}'.format(
        query_row.get('title'), query_idx, query_lang, query_genres))

    # ── Recent compound suggestions for new/modern era ─────────────────────────
    # Skip compound on refine (results cached from first call) 
    gemini_meta_cache = {}
    if era in ('new', 'modern') and not refine:
        lang_label = {
            'hi': 'Hindi', 'ko': 'Korean', 'ta': 'Tamil', 'te': 'Telugu',
            'ml': 'Malayalam', 'en': 'English', 'fr': 'French',
            'ja': 'Japanese', 'es': 'Spanish',
        }.get(query_lang, query_lang)
        genres_str = ', '.join(_get_genres(query_row))
        yr_from    = 2022 if era == 'new' else 2018
        yr_to      = CURRENT_YEAR if era == 'new' else 2021
        overview_s = str(query_row.get('overview', '') or '')[:150]

        # Emotional axis compound prompt — same architecture as mega prompt
        compound_prompt = (
            'You are an emotional film similarity engine.\n'
            'Find {lang} films from {yf}-{yt} that match the emotional architecture of "{title}".\n\n'
            'Reference film overview: {story}\n\n'
            'Profile "{title}" on these axes (0-10):\n'
            '1. Psychological Complexity\n'
            '2. Darkness\n'
            '3. Emotional Warmth\n'
            '4. Reality Instability\n'
            '5. Emotional Intensity\n'
            '6. Narrative Scope (intimate=0, epic=10)\n\n'
            'Then find {n} {lang} films from {yf}-{yt} with similar axis profiles.\n'
            'IMDB 7.0+. Match emotional tone, not genre label.\n'
            'Return JSON array: [{{"title":"","year":0,"emotional_match":"one sentence"}}]'
        ).format(lang=lang_label, yf=yr_from, yt=yr_to,
                 title=query_row.get('title'), story=overview_s, n=12)

        compound_films = compound_suggest_with_meta(compound_prompt, query_lang, n=12)
        for film in compound_films:
            film_year = int(film.get('year') or 0)
            if yr_from <= film_year <= yr_to:
                t = film.get('title', '')
                if t:
                    tc = re.sub(r'[^a-z0-9 ]', '', t.lower().strip())
                    gemini_meta_cache[tc] = film
        print('[vibe] Compound cache ({}-{}): {} films'.format(yr_from, yr_to, len(gemini_meta_cache)))

    # ── Main Groq suggestions ─────────────────────────────────────────────────
    n_suggest   = max(top_n * 2, 15)   # was top_n*4 — fewer titles = faster DB resolution
    groq_titles = groq_suggest_and_score(
        query_title    = query_row.get('title', query_title),
        query_row      = query_row,
        n              = n_suggest,
        exclude_titles = exclude_titles,
        era            = era,
    )

    # Merge: compound first, then groq (deduped)
    suggested = []
    seen_tc   = set()
    compound_orig = [f.get('title') for f in gemini_meta_cache.values() if f.get('title')]
    for t in compound_orig + groq_titles:
        if not t:
            continue
        tc = re.sub(r'[^a-z0-9 ]', '', t.lower().strip())
        if tc not in seen_tc:
            seen_tc.add(tc)
            suggested.append(t)

    # Era pre-filter on already-known DB titles
    if era:
        era_min, era_max = get_era_range(era)
        filtered = []
        for t in suggested:
            tc = re.sub(r'[^a-z0-9 ]', '', t.lower().strip())
            if tc in gemini_meta_cache:
                filtered.append(t)
                continue
            if tc in title_to_idx:
                idx_c = title_to_idx[tc]
                if idx_c < len(df):
                    yr = int(df.iloc[idx_c].to_dict().get('year', 0) or 0)
                    if (not era_min or yr >= era_min) and (not era_max or yr <= era_max):
                        filtered.append(t)
            else:
                filtered.append(t)  # unknown — let resolve() check it
        suggested = filtered
        print('[vibe] After pre-filter ({}/{}): {} titles'.format(era_min, era_max, len(suggested)))

    # ── Resolve each suggestion ───────────────────────────────────────────────
    rows    = []
    already = {_normalize_title(t) for t in (exclude_titles or [])}
    already.add(query_clean)

    for title in suggested:
        title_c = re.sub(r'[^a-z0-9 ]', '', title.lower().strip())
        if title_c in already:
            continue

        row = None

        idx = resolve(title, save_on_inject=False)
        if idx is not None:
            dupes = df[df['title_clean'] == title_c]
            if len(dupes) > 1:
                same_lang = dupes[dupes['original_language'] == query_lang] if 'original_language' in dupes.columns else pd.DataFrame()
                pool      = same_lang if not same_lang.empty else dupes
                best_idx  = pool['vote_count'].idxmax() if 'vote_count' in pool.columns else pool.index[0]
                row       = safe_row(best_idx)
            else:
                row = safe_row(idx)
        elif title_c in gemini_meta_cache:
            meta = gemini_meta_cache[title_c].copy()
            meta.setdefault('title', title)
            meta.setdefault('original_language', query_lang)
            new_idx = inject_movie(meta, save=False)
            if new_idx is not None:
                row = safe_row(new_idx)
        else:
            tmdb_data = tmdb_search(title)
            if 'error' not in tmdb_data:
                new_idx = inject_movie(tmdb_data, save=False)
                if new_idx is not None:
                    row = safe_row(new_idx)

        if row is None:
            already.add(title_c)
            continue

        # ── Unified validation gate ───────────────────────────────────────────
        ok, reason = _passes_quality(row, era, query_genres)
        if not ok:
            print('[vibe] Filtered "{}": {}'.format(row.get('title'), reason))
            already.add(title_c)
            continue

        if not _passes_era(row, era):
            print('[vibe] Wrong era "{}": year={}'.format(row.get('title'), row.get('year')))
            already.add(title_c)
            continue

        rows.append(row)
        already.add(re.sub(r'[^a-z0-9 ]', '', str(row.get('title', '')).lower().strip()))
        print('[vibe] Added: "{}" ({}) ★{:.1f}'.format(
            row.get('title'), int(row.get('year', 0) or 0), _safe_rating(row)))

        if len(rows) >= top_n * 3:  # collect up to 3x for ranker
            break

    print('[vibe] Groq pool: {} films'.format(len(rows)))

    # ── Post-resolve language filter on Groq pool ────────────────────────────
    # 8b model often ignores language rules in the prompt — filter here as safety net.
    if query_lang and query_lang != 'en' and 'original_language' in df.columns:
        ql       = str(query_lang).lower().strip()
        ql_group = _LANG_GROUPS.get(ql, 'other')

        def _lang_ok(r):
            rl = str(r.get('original_language') or '').lower().strip()
            if not rl or rl in ('nan', 'none'):
                return True   # unknown — keep it
            if rl == ql or rl == 'en':
                return True   # exact match or English always OK
            return _LANG_GROUPS.get(rl, 'other') == ql_group  # same group OK

        lang_filtered = [r for r in rows if _lang_ok(r)]
        if len(lang_filtered) >= max(top_n // 2, 2):
            removed = len(rows) - len(lang_filtered)
            if removed:
                print('[vibe] Lang post-filter: removed {} wrong-language films'.format(removed))
            rows = lang_filtered
        else:
            print('[vibe] Lang post-filter: too few after filter ({}) — keeping full pool'.format(
                len(lang_filtered)))

    # TF-IDF candidate pool — always 50 films, Groq ranks them
    try:
        tfidf_pool = tfidf_fallback(
            query_idx=query_idx, query_clean=query_clean,
            exclude_titles=list(already), era=era,
            genres_exclude=genres_exclude, genres_require=genres_require,
            top_n=50, query_genres=query_genres, query_lang=query_lang,
        )
        pool_rows = []
        for _, row in tfidf_pool.iterrows():
            rd = row.to_dict()
            rc = re.sub(r'[^a-z0-9 ]', '', str(rd.get('title', '')).lower().strip())
            if rc in already:
                continue
            ok, _ = _passes_quality(rd, era, query_genres)
            if ok and _passes_era(rd, era):
                pool_rows.append(rd)

        # Fix 3: if era pool is too small, expand with era-relaxed TF-IDF
        # so Groq has enough candidates to find good emotional matches
        if era and len(pool_rows) < 15:
            print('[vibe] Era pool too small ({}), expanding without era...'.format(len(pool_rows)))
            extra_pool = tfidf_fallback(
                query_idx=query_idx, query_clean=query_clean,
                exclude_titles=list(already), era=None,  # no era filter
                genres_exclude=genres_exclude, genres_require=genres_require,
                top_n=50, query_genres=query_genres, query_lang=query_lang,
            )
            existing_keys = {re.sub(r'[^a-z0-9 ]', '', str(r.get('title','')).lower().strip())
                             for r in pool_rows}
            for _, row in extra_pool.iterrows():
                rd = row.to_dict()
                rc = re.sub(r'[^a-z0-9 ]', '', str(rd.get('title', '')).lower().strip())
                if rc in already or rc in existing_keys:
                    continue
                ok, _ = _passes_quality(rd, None, query_genres)
                if ok:
                    pool_rows.append(rd)
                    existing_keys.add(rc)
                    if len(pool_rows) >= 40:
                        break
            print('[vibe] Expanded pool: {} candidates'.format(len(pool_rows)))

        print('[vibe] TF-IDF pool: {} candidates'.format(len(pool_rows)))
    except Exception as e:
        print('[vibe] TF-IDF pool error: {}'.format(e))
        pool_rows = []

    # Merge Groq + TF-IDF (deduped), Groq suggestions first
    combined = list(rows)
    combined_keys = {re.sub(r'[^a-z0-9 ]', '', str(r.get('title', '')).lower().strip()) for r in combined}
    for rd in pool_rows:
        key = re.sub(r'[^a-z0-9 ]', '', str(rd.get('title', '')).lower().strip())
        if key not in combined_keys:
            combined.append(rd)
            combined_keys.add(key)

    # Fix 2: Pool enrichment — resolve any Groq-suggested titles not yet in pool.
    # groq_suggest_and_score returns titles; some may not be in TF-IDF space
    # (e.g. older classics for a modern-era query). Resolve them from DB/TMDB
    # and inject into combined so Groq ranker can actually pick them.
    for t in groq_titles:
        tc = re.sub(r'[^a-z0-9 ]', '', t.lower().strip())
        if tc in combined_keys or tc in already:
            continue
        try:
            idx = resolve(t, save_on_inject=False)
            if idx is not None:
                rd = safe_row(idx)
                if rd:
                    ok, _ = _passes_quality(rd, None, query_genres)
                    if ok:
                        combined.append(rd)
                        combined_keys.add(tc)
                        print('[vibe] Pool enriched: "{}"'.format(t))
        except Exception:
            pass

    print('[vibe] Combined pool: {} candidates'.format(len(combined)))

    # Groq ranks the pool emotionally — can only pick from provided list
    # Prioritise: Groq-suggested rows first, then TF-IDF pool
    # This ensures stylistically matched films (Django, Kill Bill for IB)
    # aren't pushed out of the ranker window by keyword-matched films (WWII)
    groq_keys = {re.sub(r'[^a-z0-9 ]', '', str(r.get('title','')).lower().strip()) for r in rows}
    tfidf_only = [r for r in pool_rows if re.sub(r'[^a-z0-9 ]', '', str(r.get('title','')).lower().strip()) not in groq_keys]
    ranked_input = (rows + tfidf_only)[:30]

    if len(ranked_input) >= 2:
        try:
            ranked = groq_rank_candidates(
                query_title=query_row.get('title', query_title),
                query_row=query_row,
                candidate_rows=ranked_input,
                top_n=top_n * 2,   # ask for 2× so MMR has room to diversify
                era=era,
            )
            rows = ranked if len(ranked) >= 2 else ranked_input[:top_n * 2]
        except Exception as e:
            print('[vibe] Ranker error: {}'.format(e))
            rows = ranked_input[:top_n * 2]
    else:
        rows = combined[:top_n * 2]

    # ── MMR diversity pass ────────────────────────────────────────────────────
    # Prevents 5 near-identical films. 60% relevance / 40% diversity.
    try:
        rows = _mmr(
            selected_rows      = [],
            candidate_rows     = rows,
            embed_matrix_local = embed_matrix,
            title_to_pos       = title_to_idx,
            lambda_            = 0.6,
            top_n              = top_n,
        )
        print('[vibe] After MMR: {}'.format([r.get('title') for r in rows]))
    except Exception as e:
        print('[vibe] MMR error (skipping): {}'.format(e))
        rows = rows[:top_n]

    if not rows:
        persist_if_new(_size_before)
        return None, "No matching films found for '{}' with era={}. Try a different era.".format(
            query_title, era or 'any')

    # ── Native language guarantee ─────────────────────────────────────────────
    # If query is non-English (e.g. Hindi) but final results are mostly English,
    # inject top-rated same-language films to guarantee minimum native representation.
    # This handles cases where FAISS similarity scores favour English films.
    if query_lang and str(query_lang).lower().strip() not in ('en', ''):
        ql       = str(query_lang).lower().strip()
        min_native = max(2, top_n // 2)   # at least half should be native language

        native_count = sum(
            1 for r in rows
            if str(r.get('original_language', '')).lower().strip() == ql
        )
        print('[vibe] Native lang ({}) count in results: {}/{}'.format(ql, native_count, len(rows)))

        if native_count < min_native:
            needed = min_native - native_count
            already_titles = {re.sub(r'[^a-z0-9 ]', '', str(r.get('title', '')).lower().strip())
                              for r in rows}
            excl_set       = {re.sub(r'[^a-z0-9 ]', '', t.lower().strip())
                              for t in (exclude_titles or [])}

            # Pull top-rated same-language films from df, respecting era
            df_lang = df['original_language'].astype(str).str.lower().str.strip()
            native_pool = df[df_lang == ql].copy()
            if era:
                yr_min, yr_max = get_era_range(era)
                _ny = pd.to_numeric(native_pool['year'], errors='coerce').fillna(0)
                if yr_min: native_pool = native_pool[_ny >= yr_min]
                if yr_max: native_pool = native_pool[_ny <= yr_max]

            native_pool = native_pool[
                ~native_pool['title_clean'].isin(already_titles | excl_set)
            ]
            # weighted_score only exists on scored candidates, not raw df
            # Use vote_average * log(vote_count) as a simple proxy for quality sort
            import numpy as _np2
            _va  = pd.to_numeric(native_pool['vote_average'],  errors='coerce').fillna(0)
            _vc  = pd.to_numeric(native_pool['vote_count'],    errors='coerce').fillna(1).clip(lower=1)
            native_pool = native_pool.assign(
                _nat_score = _va * _np2.log10(_vc)
            ).sort_values('_nat_score', ascending=False)

            injected = 0
            for _, row_s in native_pool.iterrows():
                if injected >= needed:
                    break
                rd = row_s.to_dict()
                rows.append(rd)
                injected += 1
                print('[vibe] Native inject: "{}" ({})'.format(rd.get('title'), ql))

            # Re-sort: native films get a score bump so they don't all sink to bottom
            def _rescore(r):
                rl = str(r.get('original_language', '')).lower().strip()
                base = float(r.get('weighted_score') or r.get('final_score') or 0)
                return base + (0.3 if rl == ql else 0.0)
            rows = sorted(rows, key=_rescore, reverse=True)
            rows = rows[:top_n]
            print('[vibe] After native guarantee: {}'.format([r.get('title') for r in rows]))

    # ── Build output ──────────────────────────────────────────────────────────
    final    = pd.DataFrame(rows[:top_n])
    out_cols = ['title', 'year', 'genres', 'vote_average', 'overview', 'director', 'poster_url']
    out_cols = [c for c in out_cols if c in final.columns]
    final    = final[out_cols].copy()
    # Ensure JSON-serializable types (year as int, NaN as None)
    if 'year' in final.columns:
        final['year'] = pd.to_numeric(final['year'], errors='coerce').fillna(0).astype(int)
    if 'vote_average' in final.columns:
        final['vote_average'] = pd.to_numeric(final['vote_average'], errors='coerce').fillna(0).round(1)
    final = final.where(pd.notnull(final), None)

    print('[vibe] Final: {}'.format(list(final['title'].values)))

    existing_shown = LAST_SESSION.get('shown', []) if refine else []
    LAST_SESSION.update({
        'tool':  'vibe',
        'query': query_title,
        'shown': (existing_shown + list(final['title'].values))[-_MAX_SESSION_SHOWN:],
        'era':   era,
        'top_n': top_n,
        'year':  year,   # save resolved year so Show More doesn't re-disambiguate
    })

    persist_if_new(_size_before)
    result = final.to_dict('records'), None
    if not refine:
        # Evict oldest entry if cache is full
        from .shared import _VIBE_CACHE_MAX
        if len(_VIBE_CACHE) >= _VIBE_CACHE_MAX:
            oldest = next(iter(_VIBE_CACHE))
            del _VIBE_CACHE[oldest]
        _VIBE_CACHE[cache_key] = result  # v12
    return result