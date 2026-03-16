# frontend/app.py
# Run with: streamlit run app.py

import streamlit as st
import requests
import random
import os

API_URL = os.environ.get("API_URL", "http://localhost:8000")
st.set_page_config(
    page_title="CineAgent — Your Personal Film Oracle",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dramatic copy ──────────────────────────────────────────────────────────────
OPENING_LINES = [
    "The lights dim. The screen flickers. What story calls to you tonight?",
    "Somewhere out there is a film that will wreck you. Let's find it.",
    "The popcorn's ready. The projector hums. What world do you want to enter?",
    "A thousand stories. One perfect night. Where do we begin?",
]

TOOL_INTROS = {
    "vibe":    "🎭 Based on your taste, the oracle speaks...",
    "mood":    "🌙 The mood is set. The films await...",
    "compare": "⚔️  Two titans enter the ring. Only one leaves your watchlist.",
    "gems":    "💎 Hidden in the vaults of cinema, forgotten by most...",
    "llm":     "🎬 The oracle has consulted the archives...",
}

ERA_OPTIONS   = ["Any Era", "Classic", "90s", "2000s", "2010s", "Modern", "New"]
GENRE_OPTIONS = ["Any Genre","Action","Adventure","Animation","Comedy","Crime",
                 "Documentary","Drama","Family","Fantasy","History","Horror",
                 "Music","Mystery","Romance","Science Fiction","Thriller","War","Western"]

# ── Styles ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

html, body, [class*="css"] { background-color: #080808; color: #e8e0d0; font-family: 'DM Sans', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }

::-webkit-scrollbar { width: 3px; }
::-webkit-scrollbar-thumb { background: #c9a84c; border-radius: 2px; }

.cine-title {
    font-family: 'Bebas Neue', cursive; font-size: 4.5rem; letter-spacing: 0.2em;
    background: linear-gradient(135deg, #8a6a1a 0%, #c9a84c 30%, #f0d080 55%, #c9a84c 80%, #8a6a1a 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    line-height: 1; filter: drop-shadow(0 0 30px #c9a84c44); margin-bottom: 0;
}
.cine-tagline {
    font-family: 'Playfair Display', serif; font-style: italic;
    font-size: 0.9rem; color: #555; margin-top: 0.2rem; margin-bottom: 1rem;
}
.divider { border: none; border-top: 1px solid #1a1a1a; margin: 0.8rem 0 1.2rem 0; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #080808 !important; border-bottom: 1px solid #1a1a1a !important; gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important; border: none !important;
    color: #444 !important; font-family: 'DM Sans', sans-serif !important;
    font-size: 0.72rem !important; letter-spacing: 0.12em !important;
    text-transform: uppercase !important; padding: 0.6rem 1.2rem !important;
    border-bottom: 2px solid transparent !important; transition: all 0.2s !important;
}
.stTabs [aria-selected="true"] {
    color: #c9a84c !important; border-bottom: 2px solid #c9a84c !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab"]:hover { color: #888 !important; }
.stTabs [data-baseweb="tab-panel"] { padding: 1.5rem 0 0 0 !important; background: transparent !important; }

/* Panel header */
.panel-header {
    background: linear-gradient(160deg, #0f0f0f, #0a0a0a);
    border: 1px solid #1a1a1a; border-radius: 10px; padding: 1.2rem 1.4rem; margin-bottom: 1.4rem;
}
.panel-title { font-family: 'Bebas Neue', cursive; font-size: 1.8rem; letter-spacing: 0.15em; color: #c9a84c; margin-bottom: 0.2rem; }
.panel-desc  { font-family: 'Playfair Display', serif; font-style: italic; font-size: 0.82rem; color: #444; }

.lbl { font-size: 0.62rem; letter-spacing: 0.2em; text-transform: uppercase; color: #555; font-family: 'DM Sans', sans-serif; margin-bottom: 0.25rem; margin-top: 0.7rem; display: block; }

/* Inputs */
.stTextInput input, .stNumberInput input {
    background: #0a0a0a !important; border: 1px solid #1e1e1e !important;
    border-radius: 6px !important; color: #e8e0d0 !important;
    font-family: 'DM Sans', sans-serif !important; font-size: 0.9rem !important;
}
.stTextInput input:focus { border-color: #c9a84c55 !important; box-shadow: 0 0 0 2px #c9a84c11 !important; }
.stTextInput input::placeholder { color: #2e2e2e !important; font-style: italic !important; }
div[data-baseweb="select"] > div { background: #0a0a0a !important; border-color: #1e1e1e !important; color: #e8e0d0 !important; }
div[data-baseweb="select"] li { background: #0f0f0f !important; color: #c8c0b0 !important; }

/* Buttons */
.stButton button {
    background: linear-gradient(135deg, #1c1500, #2a2000) !important;
    border: 1px solid #c9a84c55 !important; color: #c9a84c !important;
    font-family: 'DM Sans', sans-serif !important; font-size: 0.72rem !important;
    letter-spacing: 0.15em !important; text-transform: uppercase !important;
    border-radius: 6px !important; transition: all 0.2s !important; font-weight: 500 !important;
    width: 100% !important;
}
.stButton button:hover { border-color: #c9a84c99 !important; box-shadow: 0 0 20px #c9a84c22 !important; }
.stFormSubmitButton button {
    background: linear-gradient(135deg, #1c1500, #2a2000) !important;
    border: 1px solid #c9a84c55 !important; color: #c9a84c !important;
    font-family: 'DM Sans', sans-serif !important; font-size: 0.72rem !important;
    letter-spacing: 0.15em !important; text-transform: uppercase !important;
    border-radius: 6px !important; transition: all 0.2s !important;
}

/* Chat */
.msg-user { display: flex; justify-content: flex-end; margin: 0.8rem 0; }
.msg-user .bubble {
    background: linear-gradient(135deg, #1c1500, #2a1f00); border: 1px solid #c9a84c44;
    border-radius: 16px 16px 2px 16px; padding: 0.65rem 1.1rem; max-width: 60%;
    font-size: 0.9rem; color: #f0d080;
}
.msg-bot { display: flex; justify-content: flex-start; margin: 0.8rem 0; align-items: flex-start; }
.msg-bot .bubble {
    background: #111; border: 1px solid #1e1e1e; border-radius: 16px 16px 16px 2px;
    padding: 0.65rem 1.1rem; max-width: 92%; font-size: 0.87rem; color: #c8c0b0; line-height: 1.7;
}
.msg-avatar {
    width: 28px; height: 28px; border-radius: 50%;
    background: linear-gradient(135deg, #1a1400, #c9a84c22); border: 1px solid #c9a84c44;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.8rem; margin-right: 0.5rem; flex-shrink: 0; margin-top: 2px;
}

.tool-badge {
    display: inline-block; font-size: 0.57rem; font-weight: 500;
    letter-spacing: 0.2em; text-transform: uppercase; padding: 2px 8px;
    border-radius: 20px; margin-bottom: 0.4rem; font-family: 'DM Sans', sans-serif;
}
.badge-vibe    { background:#0d0d1f; color:#7a9fff; border:1px solid #2a2a5a; }
.badge-mood    { background:#0d1f0d; color:#7fcf9f; border:1px solid #1a4a1a; }
.badge-compare { background:#1f0d0d; color:#ff9f7f; border:1px solid #4a1a1a; }
.badge-gems    { background:#1f1f0d; color:#dfcf7f; border:1px solid #4a4a1a; }
.badge-llm     { background:#0d1f1f; color:#7fcfdf; border:1px solid #1a4a4a; }

/* Movie cards */
.movies-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(230px, 1fr)); gap: 0.8rem; margin-top: 0.8rem; }
.movie-card {
    background: linear-gradient(160deg,#141414,#0f0f0f); border: 1px solid #1e1e1e;
    border-radius: 8px; padding: 1rem; transition: all 0.25s; position: relative; overflow: hidden;
}
.movie-card:hover { border-color: #c9a84c33; transform: translateY(-2px); box-shadow: 0 8px 25px #00000055; }
.movie-title   { font-family:'Playfair Display',serif; font-size:1rem; font-weight:700; color:#f0e8d0; margin-bottom:0.15rem; line-height:1.3; }
.movie-meta    { font-size:0.68rem; color:#555; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:0.4rem; }
.movie-rating  { display:inline-block; background:linear-gradient(135deg,#1c1500,#2a1f00); border:1px solid #c9a84c33; color:#c9a84c; font-size:0.66rem; padding:1px 6px; border-radius:10px; margin-bottom:0.4rem; }
.movie-overview{ font-size:0.76rem; color:#777; line-height:1.6; font-style:italic; display:-webkit-box; -webkit-line-clamp:3; -webkit-box-orient:vertical; overflow:hidden; }
.movie-genres  { margin-top:0.5rem; display:flex; flex-wrap:wrap; gap:3px; }
.genre-pill    { font-size:0.57rem; letter-spacing:0.08em; text-transform:uppercase; padding:1px 5px; border-radius:8px; background:#1a1a1a; color:#444; border:1px solid #222; }

/* Compare */
.compare-wrap { display:grid; grid-template-columns:1fr auto 1fr; gap:1rem; align-items:start; margin-top:0.8rem; }
.vs-div  { font-family:'Bebas Neue',cursive; font-size:1.8rem; color:#c9a84c; text-align:center; padding-top:1.5rem; }
.cmp-card{ background:#141414; border:1px solid #1e1e1e; border-radius:8px; padding:1rem; }
.cmp-title{ font-family:'Playfair Display',serif; font-size:1.05rem; font-weight:700; color:#f0e8d0; margin-bottom:0.6rem; border-bottom:1px solid #1e1e1e; padding-bottom:0.4rem; }
.cmp-row { display:flex; justify-content:space-between; padding:0.2rem 0; font-size:0.78rem; border-bottom:1px solid #0f0f0f; }
.cmp-lbl { color:#444; font-size:0.65rem; letter-spacing:0.1em; text-transform:uppercase; }
.cmp-val { color:#c8c0b0; }

.empty-state { text-align:center; padding:3rem 2rem; }
.empty-big   { font-family:'Bebas Neue',cursive; font-size:2.2rem; letter-spacing:0.2em; color:#1a1a1a; margin-bottom:0.4rem; }
.empty-small { font-family:'Playfair Display',serif; font-style:italic; font-size:0.82rem; color:#252525; }

/* Sidebar */
section[data-testid="stSidebar"] { background:#050505 !important; border-right:1px solid #111; }
.sb-logo { font-family:'Bebas Neue',cursive; font-size:1.5rem; letter-spacing:0.2em; color:#c9a84c; }
.sb-sub  { font-family:'Playfair Display',serif; font-style:italic; font-size:0.7rem; color:#2a2a2a; }
.sb-sec  { font-size:0.58rem; letter-spacing:0.2em; text-transform:uppercase; color:#2a2a2a; margin:1rem 0 0.4rem 0; }

@keyframes fadeIn { from{opacity:0;transform:translateY(6px)} to{opacity:1;transform:translateY(0)} }
.movie-card,.msg-bot,.msg-user { animation:fadeIn 0.3s ease; }
.stSpinner>div { border-top-color:#c9a84c !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
for k, v in {
    "messages":     [],
    "last_tool":    None,
    "opening_line": random.choice(OPENING_LINES),
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── API helper ─────────────────────────────────────────────────────────────────
def api(endpoint, payload):
    try:
        r = requests.post(f"{API_URL}/{endpoint}", json=payload, timeout=30)
        if r.status_code == 200:
            return r.json()
        # Non-200 — try to parse JSON error detail, else return plain message
        try:
            detail = r.json().get("detail", r.text[:200])
        except:
            detail = r.text[:200] or f"Server error {r.status_code}"
        return {"status": "error", "results": None, "error": detail}
    except requests.exceptions.ConnectionError:
        return {"status": "error", "results": None, "error": "Cannot reach backend. Is it running on port 8000?"}
    except Exception as e:
        return {"status": "error", "results": None, "error": str(e)}

# ── Renderers ──────────────────────────────────────────────────────────────────
# Poster cache — avoids re-fetching same movie repeatedly
_POSTER_CACHE = {}

def fetch_poster(title: str, year=None) -> str:
    """Fetch poster URL from backend /poster endpoint, with local cache."""
    key = f"{title}_{year}"
    if key in _POSTER_CACHE:
        return _POSTER_CACHE[key]
    try:
        params = {"title": title}
        if year and year != "—": params["year"] = int(year)
        r = requests.get(f"{API_URL}/poster", params=params, timeout=8)
        url = r.json().get("poster_url", "")
        _POSTER_CACHE[key] = url
        return url
    except:
        return ""

def render_cards(movies):
    if not movies: return
    cards = ""
    for m in movies:
        title      = m.get("title", "Unknown")
        year       = int(m.get("year", 0)) if m.get("year") else "—"
        rating     = m.get("vote_average", 0)
        overview   = m.get("overview", "No synopsis on record.")
        genres     = m.get("genres", "")
        director   = m.get("director", "")
        # Use stored poster_url or fetch from TMDB
        poster_url = m.get("poster_url", "") or ""
        if not poster_url or str(poster_url) in ("", "nan", "None"):
            poster_url = fetch_poster(title, year)
        stars = "★" * round(rating / 2) if rating else ""
        rd    = f"{stars} {rating:.1f}" if rating else "Unrated"
        pills = "".join(f'<span class="genre-pill">{g.strip()}</span>' for g in str(genres).split(",")[:3] if g.strip())
        poster_html = f'<img src="{poster_url}" style="width:100%;border-radius:4px;margin-bottom:0.6rem;display:block;object-fit:cover;max-height:320px;" />' if poster_url else '<div style="width:100%;height:160px;background:#111;border-radius:4px;margin-bottom:0.6rem;display:flex;align-items:center;justify-content:center;font-size:2.5rem;color:#222;">🎬</div>'
        dir_html    = f'<div style="font-size:0.68rem;color:#555;margin-bottom:0.3rem;letter-spacing:0.05em;">🎬 {director}</div>' if director else ""
        cards      += f'<div class="movie-card">{poster_html}<div class="movie-title">{title}</div><div class="movie-meta">{year}</div>{dir_html}<div class="movie-rating">{rd}</div><div class="movie-overview">{overview}</div><div class="movie-genres">{pills}</div></div>'
    st.markdown(f'<div class="movies-grid">{cards}</div>', unsafe_allow_html=True)

def render_compare(data):
    if not data or not isinstance(data, dict):
        st.markdown(f'<div style="color:#888;font-style:italic;">{data}</div>', unsafe_allow_html=True)
        return
    def card(m, color):
        fields = [("Year",m.get("year","—")),("Rating",f"{m.get('vote_average','—')} / 10"),
                  ("Votes",f"{int(m.get('vote_count',0)):,}" if m.get("vote_count") else "—"),
                  ("Genres",m.get("genres","—")),("Director",m.get("director","—"))]
        rows = "".join(f'<div class="cmp-row"><span class="cmp-lbl">{k}</span><span class="cmp-val">{v}</span></div>' for k,v in fields)
        ov = m.get("overview","")
        return f'<div class="cmp-card" style="border-top:2px solid {color}55;"><div class="cmp-title">{m.get("title","?")}</div>{rows}<div style="margin-top:0.6rem;font-size:0.74rem;color:#555;font-style:italic;line-height:1.5;">{ov[:200]}{"..." if len(ov)>200 else ""}</div></div>'
    st.markdown(f'<div class="compare-wrap">{card(data.get("movie1",{}),"#7a9fff")}<div class="vs-div">VS</div>{card(data.get("movie2",{}),"#ff9f7f")}</div>', unsafe_allow_html=True)
    verdict = data.get("verdict") or data.get("analysis") or data.get("summary")
    if verdict:
        st.markdown(f'<div style="margin-top:0.8rem;background:#0f0f0f;border:1px solid #1e1e1e;border-left:2px solid #c9a84c;border-radius:6px;padding:0.9rem 1.1rem;font-size:0.82rem;color:#a09080;line-height:1.7;font-style:italic;">{verdict}</div>', unsafe_allow_html=True)

def render_bot(tool, results, raw={}):
    badges = {"vibe":"badge-vibe","mood":"badge-mood","compare":"badge-compare","gems":"badge-gems","llm":"badge-llm"}
    labels = {"vibe":"Similar Vibes","mood":"Mood Match","compare":"Face-Off","gems":"Hidden Gems","llm":"Oracle Says"}
    badge  = f'<div class="tool-badge {badges.get(tool,"badge-llm")}">{labels.get(tool,tool.upper())}</div>'
    intro  = TOOL_INTROS.get(tool, "🎬 The oracle has spoken...")

    if tool == "compare":
        st.markdown(f'<div class="msg-bot"><div class="msg-avatar">🎬</div><div class="bubble">{badge}<div style="font-style:italic;color:#555;font-size:0.76rem;">{intro}</div></div></div>', unsafe_allow_html=True)
        render_compare(results)
        return
    if isinstance(results, list) and results:
        st.markdown(f'<div class="msg-bot"><div class="msg-avatar">🎬</div><div class="bubble">{badge}<div style="font-style:italic;color:#555;font-size:0.76rem;">{intro}</div></div></div>', unsafe_allow_html=True)
        render_cards(results)
        return
    if isinstance(results, str) and results:
        content = f'<div style="white-space:pre-wrap;line-height:1.7;">{results}</div>'
    else:
        err = raw.get("error") or raw.get("message") or raw.get("detail") or "No results. Try different parameters."
        content = f'<div style="color:#c06060;font-style:italic;">{err}</div>'
    st.markdown(f'<div class="msg-bot"><div class="msg-avatar">🎬</div><div class="bubble">{badge}<div style="font-style:italic;color:#555;font-size:0.76rem;margin-bottom:0.4rem;">{intro}</div>{content}</div></div>', unsafe_allow_html=True)

def push(label, tool, results, raw):
    st.session_state.messages.append({"role":"user","content":label})
    st.session_state.messages.append({"role":"assistant","tool":tool,"results":results,"raw":raw})
    st.session_state.last_tool = tool

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="sb-logo">🎬 CINEAGENT</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-sub">Your Personal Film Oracle</div>', unsafe_allow_html=True)
    st.markdown('<hr style="border-color:#111;margin:0.8rem 0;">', unsafe_allow_html=True)

    if st.session_state.last_tool:
        st.markdown('<div class="sb-sec">Last Used</div>', unsafe_allow_html=True)
        icons = {"vibe":"🎭","mood":"🌙","compare":"⚔️","gems":"💎","llm":"🔮"}
        st.markdown(f'<div style="font-size:0.78rem;color:#444;padding:0.3rem 0;">{icons.get(st.session_state.last_tool,"")} {st.session_state.last_tool.title()}</div>', unsafe_allow_html=True)
        st.markdown('<hr style="border-color:#111;margin:0.8rem 0;">', unsafe_allow_html=True)

    if st.button("✦ Clear the Screen"):
        st.session_state.messages = []
        st.session_state.last_tool = None
        st.session_state.opening_line = random.choice(OPENING_LINES)
        try: requests.post(f"{API_URL}/llm_chat/reset", timeout=5)
        except: pass
        st.rerun()

    st.markdown('<div style="margin-top:2rem;font-size:0.55rem;color:#161616;letter-spacing:0.15em;text-transform:uppercase;">Powered by Groq + TF-IDF</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="cine-title">CINEAGENT</div>', unsafe_allow_html=True)
st.markdown('<div class="cine-tagline">Every film has a soul. We help you find yours.</div>', unsafe_allow_html=True)
st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TABS — one per tool
# ═══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎭  Vibe Match",
    "🌙  Mood Engine",
    "⚔️   Face-Off",
    "💎  Gem Hunter",
    "🔮  Oracle",
])

# ── TAB 1: VIBE MATCH ─────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="panel-header"><div class="panel-title">🎭 VIBE MATCH</div><div class="panel-desc">"Name a film you love. We\'ll find its cinematic soulmates."</div></div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([3, 1, 1])
    with c1:
        st.markdown('<span class="lbl">Movie Title — any language, any era</span>', unsafe_allow_html=True)
        v_title = st.text_input("##vt", placeholder="e.g. Inception, 3 Idiots, Parasite, Amélie...", label_visibility="collapsed")
    with c2:
        st.markdown('<span class="lbl">Release Year (optional)</span>', unsafe_allow_html=True)
        v_year = st.number_input("##vy", min_value=1900, max_value=2025, value=None, placeholder="e.g. 2010", label_visibility="collapsed", format="%d")
    with c3:
        st.markdown('<span class="lbl">Results</span>', unsafe_allow_html=True)
        v_topn = st.selectbox("##vn", [5, 8, 10, 15], label_visibility="collapsed")

    if st.button("✦ Find My Soulmates", key="vibe_go"):
        if not v_title.strip():
            st.warning("Name a film first. The oracle needs something to work with.")
        else:
            with st.spinner("Searching the cinematic multiverse..."):
                # Detect refinement phrases — send as refine= not query_title
                refine_phrases = ["more", "some more", "suggest more", "different",
                                  "different ones", "more options", "show more",
                                  "not this", "something else", "again"]
                is_refine = v_title.strip().lower() in refine_phrases or                             v_title.strip().lower().startswith("more ") or                             v_title.strip().lower().startswith("not ")
                if is_refine:
                    payload = {"query_title": v_title.strip(), "top_n": v_topn,
                               "refine": v_title.strip()}
                else:
                    payload = {"query_title": v_title.strip(), "top_n": v_topn}
                    if v_year: payload["year"] = int(v_year)
                resp = api("recommend", payload)

            if resp.get("status") == "disambig":
                # Multiple movies with same title — show options and ask for year
                options = resp.get("options", [])
                st.warning(f"Multiple films named **{v_title.strip()}** found. Pick the one you mean by entering its year:")
                for opt in options:
                    yr  = int(opt.get("year", 0))
                    ttl = opt.get("title", "")
                    gen = opt.get("genres", "")
                    rat = opt.get("vote_average", "")
                    st.markdown(f'<div style="background:#0f0f0f;border:1px solid #1e1e1e;border-radius:6px;padding:0.5rem 0.8rem;margin-bottom:0.3rem;font-size:0.82rem;color:#888;"><span style="color:#c9a84c;font-weight:600;">{yr}</span> — {ttl} <span style="color:#444;font-size:0.72rem;">({gen})</span> ★ {rat}</div>', unsafe_allow_html=True)
                st.info("Enter the release year in the **Release Year** field above and search again.")
            else:
                push(f'Films like "{v_title.strip()}"', "vibe", resp.get("results"), resp)
                st.rerun()

# ── TAB 2: MOOD ENGINE ────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="panel-header"><div class="panel-title">🌙 MOOD ENGINE</div><div class="panel-desc">"Tell us how you feel. We\'ll find exactly what you need to watch."</div></div>', unsafe_allow_html=True)
    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown('<span class="lbl">Your Mood / Feeling / Occasion / Craving</span>', unsafe_allow_html=True)
        m_mood = st.text_input("##mm", placeholder="e.g. rainy Sunday, heartbroken, girls night, 2am existential crisis...", label_visibility="collapsed")
    with c2:
        st.markdown('<span class="lbl">Era</span>', unsafe_allow_html=True)
        m_era = st.selectbox("##me", ERA_OPTIONS, label_visibility="collapsed")

    c3, c4 = st.columns(2)
    with c3:
        st.markdown('<span class="lbl">Results</span>', unsafe_allow_html=True)
        m_topn = st.selectbox("##mn", [5, 8, 10, 15], index=2, label_visibility="collapsed")
    with c4:
        st.markdown('<span class="lbl">Min Votes (higher = more popular only)</span>', unsafe_allow_html=True)
        m_votes = st.selectbox("##mv", [10, 50, 100, 500, 1000], index=2, label_visibility="collapsed")

    if st.button("✦ Set the Mood", key="mood_go"):
        if not m_mood.strip():
            st.warning("Tell us how you're feeling. The oracle is listening.")
        else:
            with st.spinner("Reading the room..."):
                payload = {"mood": m_mood.strip(), "top_n": m_topn, "min_votes": m_votes}
                if m_era != "Any Era": payload["era"] = m_era.lower()
                resp = api("mood", payload)
                tool = resp.get("tool", "mood")
                push(f'Mood: "{m_mood.strip()}"' + (f" · {m_era}" if m_era != "Any Era" else ""), tool, resp.get("results"), resp)
            st.rerun()

# ── TAB 3: FACE-OFF ───────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="panel-header"><div class="panel-title">⚔️ FACE-OFF</div><div class="panel-desc">"Two films. One verdict. The debate ends here."</div></div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<span class="lbl">First Film</span>', unsafe_allow_html=True)
        c_t1 = st.text_input("##ct1", placeholder="e.g. The Godfather", label_visibility="collapsed")
        st.markdown('<span class="lbl">Year (optional)</span>', unsafe_allow_html=True)
        c_y1 = st.number_input("##cy1", min_value=1900, max_value=2025, value=None, placeholder="e.g. 1972", label_visibility="collapsed", format="%d")
    with c2:
        st.markdown('<span class="lbl">Second Film</span>', unsafe_allow_html=True)
        c_t2 = st.text_input("##ct2", placeholder="e.g. Goodfellas", label_visibility="collapsed")
        st.markdown('<span class="lbl">Year (optional)</span>', unsafe_allow_html=True)
        c_y2 = st.number_input("##cy2", min_value=1900, max_value=2025, value=None, placeholder="e.g. 1990", label_visibility="collapsed", format="%d")

    if st.button("✦ Begin the Face-Off", key="compare_go"):
        if not c_t1.strip() or not c_t2.strip():
            st.warning("Two films must enter. Name them both.")
        else:
            with st.spinner("Summoning both films to the arena..."):
                payload = {"title1": c_t1.strip(), "title2": c_t2.strip()}
                if c_y1: payload["year1"] = int(c_y1)
                if c_y2: payload["year2"] = int(c_y2)
                resp = api("compare", payload)
                push(f'"{c_t1.strip()}" vs "{c_t2.strip()}"', "compare", resp.get("results"), resp)
            st.rerun()

# ── TAB 4: GEM HUNTER ────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="panel-header"><div class="panel-title">💎 GEM HUNTER</div><div class="panel-desc">"The films the world forgot. The ones worth remembering."</div></div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<span class="lbl">Genre</span>', unsafe_allow_html=True)
        g_genre = st.selectbox("##gg", GENRE_OPTIONS, label_visibility="collapsed")
        st.markdown('<span class="lbl">Era</span>', unsafe_allow_html=True)
        g_era = st.selectbox("##ge", ERA_OPTIONS, label_visibility="collapsed")
    with c2:
        st.markdown('<span class="lbl">Results</span>', unsafe_allow_html=True)
        g_topn = st.selectbox("##gn", [5, 8, 10, 15], index=2, label_visibility="collapsed")
        st.markdown('<span class="lbl">Min Votes</span>', unsafe_allow_html=True)
        g_votes = st.selectbox("##gv", [10, 50, 100, 200, 500], index=1, label_visibility="collapsed")

    st.markdown('<span class="lbl">Obscurity Ceiling — lower means more obscure</span>', unsafe_allow_html=True)
    g_pop = st.slider("##gp", min_value=5.0, max_value=100.0, value=30.0, step=5.0, label_visibility="collapsed")
    st.markdown(f'<div style="font-size:0.65rem;color:#333;margin-top:-0.5rem;">Max popularity: {g_pop:.0f}</div>', unsafe_allow_html=True)

    if st.button("✦ Hunt the Gems", key="gems_go"):
        with st.spinner("Digging through the vaults of forgotten cinema..."):
            payload = {"top_n": g_topn, "min_votes": g_votes, "max_popularity": g_pop}
            if g_genre != "Any Genre": payload["genre"] = g_genre
            if g_era   != "Any Era":   payload["era"]   = g_era.lower()
            resp = api("gems", payload)
            label = "Hidden gems" + (f" · {g_genre}" if g_genre != "Any Genre" else "") + (f" · {g_era}" if g_era != "Any Era" else "")
            push(label, "gems", resp.get("results"), resp)
        st.rerun()

# ── TAB 5: ORACLE ─────────────────────────────────────────────────────────────
with tab5:
    st.markdown('<div class="panel-header"><div class="panel-title">🔮 THE ORACLE</div><div class="panel-desc">"Directors. Actors. Awards. Complex queries. The oracle knows everything about cinema."</div></div>', unsafe_allow_html=True)
    st.markdown('<span class="lbl">Ask Anything About Cinema</span>', unsafe_allow_html=True)

    with st.form("oracle_form", clear_on_submit=True):
        o_msg = st.text_input("##oq", placeholder="e.g. Best Villeneuve films ranked by how haunting they are... Who won Best Picture in 2020?", label_visibility="collapsed")
        c1, c2 = st.columns([4, 1])
        with c1: o_go    = st.form_submit_button("✦ Consult the Oracle", use_container_width=True)
        with c2: o_reset = st.form_submit_button("↺ Reset", use_container_width=True)

    if o_reset:
        try: requests.post(f"{API_URL}/llm_chat/reset", timeout=5)
        except: pass
        st.success("The oracle's memory has been wiped. Fresh conversation begins.")

    if o_go and o_msg.strip():
        with st.spinner("The oracle is consulting the archives of all cinema..."):
            resp = api("llm_chat", {"message": o_msg.strip()})
            push(o_msg.strip(), "llm", resp.get("results"), resp)
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# RESULTS FEED
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<hr class="divider">', unsafe_allow_html=True)

if not st.session_state.messages:
    st.markdown(f'<div class="empty-state"><div class="empty-big">THE SCREEN AWAITS</div><div class="empty-small">"{st.session_state.opening_line}"</div></div>', unsafe_allow_html=True)
else:
    st.markdown('<div style="font-size:0.55rem;letter-spacing:0.25em;text-transform:uppercase;color:#1e1e1e;margin-bottom:1rem;">— Results —</div>', unsafe_allow_html=True)
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="msg-user"><div class="bubble">{msg.get("content","")}</div></div>', unsafe_allow_html=True)
        else:
            render_bot(msg.get("tool","llm"), msg.get("results"), msg.get("raw",{}))
