# backend/tools/agent.py
# ═══════════════════════════════════════════════════════════════════════════════
# AGENT — routes any user message to the right tool
# ═══════════════════════════════════════════════════════════════════════════════

import re
import json
from groq import Groq

from .shared    import df, GROQ_API_KEY, LAST_SESSION

# Shared Groq client — instantiated once per server session
_groq_client = None
def _get_groq():
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client
from .vibe      import recommend_by_vibe
from .mood      import mood_based_recommend
from .compare   import compare_movies
from .gems      import discover_hidden_gems
from .llm_chat  import llm_movie_chat


def groq_route_message(message: str, last_session: dict) -> dict:
    """
    Groq reads the user message and decides which tool to call.

    Tool decision priority:
    - Non-English titles (Hindi, Korean, French, etc.) are still valid movies → tool=vibe
    - '3 idiots', 'dangal', 'RRR', 'parasite' etc. are all valid query_title values
      vibe      — "something like [movie]", "similar to [movie]"
      mood      — mood/feeling/occasion/setting words, genre-only queries
      compare   — "vs", "compare", "better than"
      gems      — "hidden gem", "underrated", "obscure"
      llm       — everything else that is movie-related:
                  director/crew/actor questions, year+genre combos,
                  complex filters, trivia, awards, plot questions,
                  follow-up questions on previous llm answers, etc.
      off_topic — anything NOT about movies/cinema
    """
    session_context = (
        f"Previous tool: {last_session.get('tool')}\n"
        f"Previous query: {last_session.get('query')}\n"
        f"Shown titles: {last_session.get('shown', [])}\n"
    ) if last_session.get('tool') else "No previous session.\n"

    prompt = (
        f"You are a movie recommendation agent. A user sent this message:\n"
        f"'{message}'\n\n"
        f"Session context:\n{session_context}\n"
        f"Decide what to do and return ONLY a JSON object:\n"
        f"{{\n"
        f'  "tool": "vibe" | "mood" | "compare" | "gems" | "llm" | "off_topic",\n'
        f'  "params": {{\n'
        f'    "query_title": "title if tool=vibe, else null",\n'
        f'    "year":        "release year if mentioned, else null",\n'
        f'    "mood":        "mood string if tool=mood, else null",\n'
        f'    "era":         "90s|2000s|2010s|modern|new|classic if mentioned, else null",\n'
        f'    "title1":      "first title if tool=compare, else null",\n'
        f'    "title2":      "second title if tool=compare, else null",\n'
        f'    "year1":       "release year of title1 if mentioned, else null",\n'
        f'    "year2":       "release year of title2 if mentioned, else null",\n'
        f'    "genre":       "genre if tool=gems, else null",\n'
        f'    "refine":      "refinement string if user wants changes to previous results, else null"\n'
        f'  }},\n'
        f'  "message": "friendly message if off_topic"\n'
        f"}}\n\n"
        f"Routing rules (follow in order):\n"
        f"1. 'like [movie]', 'similar to [movie]', 'suggest something like X' → tool=vibe\n"
        f"2. mood/feeling/occasion/setting/genre-only (no title, no director) → tool=mood\n"
        f"3. 'vs', 'compare', 'better than', 'difference between' → tool=compare\n"
        f"4. 'hidden gem', 'underrated', 'obscure' → tool=gems\n"
        f"5. ANYTHING else that is movie-related → tool=llm\n"
        f"   This includes: director/crew/actor queries, year+genre combos,\n"
        f"   'thrillers from 2020s', 'Nolan movies', 'best 90s horror',\n"
        f"   'who directed X', 'movies with Y actor', plot questions, trivia,\n"
        f"   awards, box office, follow-up questions — ALL go to llm.\n"
        f"6. Anything NOT about movies/cinema → tool=off_topic\n\n"
        f"CRITICAL: When no query_title is present and a year or director is mentioned,\n"
        f"ALWAYS route to llm, never to vibe.\n"
        f"ONLY set refine if user explicitly says 'more', 'different', 'not slow', 'suggest again'.\n"
        f"A new mood/occasion is ALWAYS a fresh mood call, NOT a refinement.\n"
    )
    try:
        resp = _get_groq().chat.completions.create(
            model='llama-3.3-70b-versatile',
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0.1,
            max_tokens=400,
        )
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r'^```json?\s*', '', raw, flags=re.MULTILINE)
        raw = re.sub(r'```\s*$',      '', raw, flags=re.MULTILINE)
        return json.loads(raw.strip())
    except Exception as e:
        print(f"[agent] Routing error: {e}")
        return {'tool': 'off_topic', 'message': 'Sorry, something went wrong.'}


def run_agent(message: str) -> dict:
    """
    Master agent — routes message to correct tool and returns results.
    Called by FastAPI /vibe endpoint.
    """
    print(f"\n[agent] User: '{message}'")

    route  = groq_route_message(message, LAST_SESSION)
    tool   = route.get('tool')
    params = route.get('params', {})

    print(f"[agent] Routed to: {tool}")
    print(f"[agent] Params: {params}")

    # ── vibe ──────────────────────────────────────────────────────────────────
    if tool == 'vibe':
        # Safety: no title → fall through to llm
        if not params.get('query_title'):
            print("[agent] vibe routed but no query_title — falling back to llm")
            result, err = llm_movie_chat(message)
            return {'tool': 'llm', 'results': result, 'error': err}

        refine = params.get('refine')
        if refine:
            result, err = recommend_by_vibe(
                params.get('query_title', ''),
                refine=refine,
            )
        else:
            raw_year = params.get('year')
            result, err = recommend_by_vibe(
                query_title = params.get('query_title', ''),
                year        = int(raw_year) if raw_year else None,
            )
        return {'tool': 'vibe', 'results': result, 'error': err}

    # ── mood ──────────────────────────────────────────────────────────────────
    elif tool == 'mood':
        refine = params.get('refine')
        if refine:
            result, err = mood_based_recommend(
                mood   = params.get('mood', ''),
                refine = refine,
            )
        else:
            result, err = mood_based_recommend(
                mood = params.get('mood', ''),
                era  = params.get('era'),
            )
        # Mood tool signals LLM fallback when confidence is low or no genre match
        if err == 'LLM_FALLBACK':
            print(f"[agent] Mood fallback → llm_chat for '{message}'")
            result, err = llm_movie_chat(message)
            return {'tool': 'llm', 'results': result, 'error': err}
        return {'tool': 'mood', 'results': result, 'error': err}

    # ── compare ───────────────────────────────────────────────────────────────
    elif tool == 'compare':
        result, err = compare_movies(
            title1 = params.get('title1', ''),
            title2 = params.get('title2', ''),
            year1  = params.get('year1'),
            year2  = params.get('year2'),
        )
        # Auto-resolve disambiguation — pick highest rated version
        if err and 'ambiguous' in err.lower():
            for title_key, year_key in [('title1', 'year1'), ('title2', 'year2')]:
                title = params.get(title_key, '')
                if not title:
                    continue
                clean   = re.sub(r'[^a-z0-9 ]', '', title.lower().strip())
                matches = df[df['title_clean'] == clean]
                if len(matches) > 1:
                    try:
                        best_year = int(matches.loc[matches['vote_average'].idxmax(), 'year'])
                    except (TypeError, ValueError):
                        continue
                    print(f"[agent] Auto-resolved '{title}' → {best_year} (highest rated)")
                    params[year_key] = best_year
            result, err = compare_movies(
                title1 = params.get('title1', ''),
                title2 = params.get('title2', ''),
                year1  = params.get('year1'),
                year2  = params.get('year2'),
            )
        # If compare still fails after disambiguation, fall back to llm
        if err and not result:
            print(f"[agent] Compare still failed after disambiguation — falling back to llm")
            result, err = llm_movie_chat(message)
            return {'tool': 'llm', 'results': result, 'error': err}
        return {'tool': 'compare', 'results': result, 'error': err}

    # ── gems ──────────────────────────────────────────────────────────────────
    elif tool == 'gems':
        raw_genre = params.get('genre')
        if raw_genre:
            raw_genre = raw_genre.strip().title()
            # Alias map handles common variants — rstrip('s') only as last resort
            _GENRE_ALIASES = {
                'Sci-Fi':           'Science Fiction',
                'Sci Fi':           'Science Fiction',
                'Scifi':            'Science Fiction',
                'Science-Fiction':  'Science Fiction',
                'Rom Com':          'Romance',
                'Romcom':           'Romance',
                'Romantic Comedy':  'Romance',
                'Animated':         'Animation',
                'Thrillers':        'Thriller',
                'Horrors':          'Horror',
                'Comedies':         'Comedy',
                'Dramas':           'Drama',
                'Westerns':         'Western',
                'Documentaries':    'Documentary',
            }
            _VALID_GENRES = {
                'Action', 'Adventure', 'Animation', 'Comedy', 'Crime',
                'Documentary', 'Drama', 'Family', 'Fantasy', 'History',
                'Horror', 'Music', 'Mystery', 'Romance', 'Science Fiction',
                'Thriller', 'War', 'Western',
            }
            raw_genre = _GENRE_ALIASES.get(raw_genre, raw_genre)
            # Only strip trailing 's' if it produces a known genre
            if raw_genre not in _VALID_GENRES:
                stripped = raw_genre.rstrip('s')
                if stripped in _VALID_GENRES:
                    raw_genre = stripped
        result, err = discover_hidden_gems(
            genre = raw_genre,
            era   = params.get('era'),
        )
        return {'tool': 'gems', 'results': result, 'error': err}

    # ── llm — pure LLM fallback for all other movie queries ──────────────────
    elif tool == 'llm':
        result, err = llm_movie_chat(message)
        return {'tool': 'llm', 'results': result, 'error': err}

    # ── off_topic ─────────────────────────────────────────────────────────────
    else:
        return {
            'tool':    'off_topic',
            'results': None,
            'error':   None,
            'message': route.get('message', "I only help with movie recommendations 🎬"),
        }