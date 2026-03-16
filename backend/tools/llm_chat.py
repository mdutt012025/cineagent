# backend/tools/llm_chat.py
# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 5: llm_movie_chat
# Pure Groq LLM with web search — handles any movie query including 2024-2026.
# ═══════════════════════════════════════════════════════════════════════════════

import re
import time as _time
from groq import Groq
from .shared import GROQ_API_KEY, TMDB_API_KEY, tmdb_search

# ── Shared Groq client — instantiated once per server session ─────────────────
_groq_client = None
def _get_client():
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client

# ── Conversation history ──────────────────────────────────────────────────────
_CHAT_HISTORY = []

SYSTEM_PROMPT = """You are CineBot, an expert movie assistant with encyclopedic knowledge of world cinema.
You have access to a web search tool — USE IT for any question about:
- Movies from 2024, 2025, or 2026
- Recent releases, box office, OTT, reviews
- Current news about films, directors, actors
- Anything where your training data might be outdated

You can handle:
- Movie recommendations with any combination of filters
- Director and crew information — including Indian directors (Anurag Kashyap, Zoya Akhtar, Mani Ratnam, S.S. Rajamouli, Sanjay Leela Bhansali etc.)
- Actor filmographies — Bollywood, South Indian, Hollywood, Korean, any industry
- Awards: Filmfare, National Film Awards, Oscars, Cannes, BAFTA, etc.
- Box office, OTT availability, trivia, behind-the-scenes facts
- Plot explanations and analysis
- Complex multi-filter queries ("best Irrfan Khan films ranked by emotional depth")

Rules:
- ONLY answer movie-related questions. If asked anything unrelated, politely decline.
- When recommending movies, always include: title, year, director, and a brief reason why.
- Be conversational, enthusiastic, and concise.
- For recommendations, aim for 5 suggestions unless asked otherwise.
- Format recommendations as a clean numbered list.
- Never make up movies that don't exist — use web search if unsure.
- You are equally knowledgeable about Hindi, Tamil, Telugu, Malayalam, Korean, Japanese, French and all world cinema — do not default to Hollywood only.
- CONTENT POLICY: Do NOT recommend explicitly sexual, erotic, or adult-only films. If asked for "sensual" or "steamy" recommendations, redirect to passionate romance suitable for general audiences (e.g. Before Sunrise, Veer-Zaara, Dil To Pagal Hai)."""


def _groq_call_with_retry(messages, use_search=True):
    """
    Call Groq with optional web search tool.
    Handles 429 rate limit with retry, tool_calls response, and plain fallback.
    Returns final text answer or None.
    """
    client = _get_client()

    def _do_call(msgs, tools=None):
        kwargs = dict(
            model='llama-3.3-70b-versatile',
            messages=msgs,
            temperature=0.4,
            max_tokens=1000,
        )
        if tools:
            kwargs['tools'] = tools
        for attempt in range(2):
            try:
                return client.chat.completions.create(**kwargs)
            except Exception as e:
                err = str(e)
                if '429' in err and attempt == 0:
                    m = re.search(r'try again in (\d+\.?\d*)s', err)
                    wait = min(int(float(m.group(1))) + 1, 15) if m else 8
                    print(f'[llm_chat] Rate limit — waiting {wait}s')
                    _time.sleep(wait)
                else:
                    raise
        return None

    # ── Try with web search ───────────────────────────────────────────────────
    if use_search:
        try:
            resp = _do_call(messages, tools=[{"type": "web_search_20250305", "name": "web_search"}])
            if resp is None:
                raise Exception("No response")

            choice = resp.choices[0]
            msg    = choice.message

            # If model triggered web search, handle tool_calls
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                print(f'[llm_chat] Web search triggered ({len(msg.tool_calls)} call(s))')
                follow_msgs = list(messages) + [{"role": "assistant", "content": msg.content or "", "tool_calls": [
                    {"id": tc.id, "type": tc.type, "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in msg.tool_calls
                ]}]
                for tc in msg.tool_calls:
                    follow_msgs.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": "Search completed."
                    })
                resp2 = _do_call(follow_msgs)
                if resp2:
                    answer = resp2.choices[0].message.content or ""
                    if answer.strip():
                        return answer.strip()

            if msg.content and msg.content.strip():
                return msg.content.strip()

        except Exception as e:
            print(f'[llm_chat] Web search call failed: {str(e)[:100]} — falling back')

    # ── Plain call fallback ───────────────────────────────────────────────────
    try:
        resp = _do_call(messages)
        if resp:
            answer = resp.choices[0].message.content or ""
            return answer.strip() or None
    except Exception as e:
        print(f'[llm_chat] Plain call also failed: {e}')

    return None


def llm_movie_chat(message: str, reset: bool = False) -> tuple:
    global _CHAT_HISTORY

    if reset:
        _CHAT_HISTORY = []
        print("[llm_chat] History cleared")
        return "History cleared.", None

    _CHAT_HISTORY.append({"role": "user", "content": message})
    history_window = _CHAT_HISTORY[-20:]

    # Inject TMDB live context for recent queries
    recent_keywords = ['2024', '2025', '2026', 'new', 'latest', 'recent',
                       'upcoming', 'just released', 'this year', 'now showing']
    is_recent = any(kw in message.lower() for kw in recent_keywords)

    system = SYSTEM_PROMPT
    if is_recent and TMDB_API_KEY and TMDB_API_KEY != 'YOUR_TMDB_KEY_HERE':
        # Only call TMDB if message looks like a specific film title
        _msg_clean = message.strip().rstrip('?').strip()
        _looks_like_title = (
            len(_msg_clean.split()) <= 6 and
            not any(w in _msg_clean.lower() for w in
                    ['what', 'who', 'when', 'where', 'how', 'best', 'top', 'list', 'recommend'])
        )
        if _looks_like_title:
            tmdb_result = tmdb_search(_msg_clean[:60])
            if 'error' not in tmdb_result:
                system += (
                    f"\n\n[TMDB Live Data] {tmdb_result['title']} ({tmdb_result.get('year')}) — "
                    f"{tmdb_result.get('overview','')[:200]} | "
                    f"Rating: {tmdb_result.get('vote_average')} | "
                    f"Director: {tmdb_result.get('director')} | "
                    f"Genres: {', '.join(tmdb_result.get('genres') or [])}"
                )

    messages = [{"role": "system", "content": system}] + history_window

    try:
        answer = _groq_call_with_retry(messages, use_search=True)
        if not answer:
            answer = "The oracle is momentarily silent. Please try again."

        _CHAT_HISTORY.append({"role": "assistant", "content": answer})
        print(f"[llm_chat] Responded ({len(answer)} chars), history={len(_CHAT_HISTORY)} turns")
        return answer, None

    except Exception as e:
        print(f"[llm_chat] Error: {e}")
        if _CHAT_HISTORY and _CHAT_HISTORY[-1]['role'] == 'user':
            _CHAT_HISTORY.pop()
        return None, f"LLM error: {e}"


def get_chat_history() -> list:
    return _CHAT_HISTORY.copy()


def clear_chat_history():
    global _CHAT_HISTORY
    _CHAT_HISTORY = []
    print("[llm_chat] History explicitly cleared")