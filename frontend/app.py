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
.block-container { padding-top: 1.8rem; padding-bottom: 3rem; max-width: 1200px; }

::-webkit-scrollbar { width: 3px; }
::-webkit-scrollbar-thumb { background: #c9a84c; border-radius: 2px; }

/* Header */
.cine-title {
    font-family: 'Bebas Neue', cursive; font-size: 5rem; letter-spacing: 0.25em;
    background: linear-gradient(135deg, #8a6a1a 0%, #c9a84c 30%, #f0d080 55%, #c9a84c 80%, #8a6a1a 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    line-height: 1; filter: drop-shadow(0 0 40px #c9a84c33); margin-bottom: 0;
}
.cine-tagline {
    font-family: 'Playfair Display', serif; font-style: italic;
    font-size: 0.95rem; color: #444; margin-top: 0.3rem; margin-bottom: 1.2rem; letter-spacing: 0.03em;
}
.divider { border: none; border-top: 1px solid #161616; margin: 0.6rem 0 1.4rem 0; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #080808 !important; border-bottom: 1px solid #161616 !important; gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important; border: none !important;
    color: #383838 !important; font-family: 'DM Sans', sans-serif !important;
    font-size: 0.7rem !important; letter-spacing: 0.14em !important;
    text-transform: uppercase !important; padding: 0.75rem 1.4rem !important;
    border-bottom: 2px solid transparent !important; transition: all 0.2s !important;
}
.stTabs [aria-selected="true"] {
    color: #c9a84c !important; border-bottom: 2px solid #c9a84c !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab"]:hover { color: #777 !important; }
.stTabs [data-baseweb="tab-panel"] { padding: 1.8rem 0 0 0 !important; background: transparent !important; }

/* Panel header */
.panel-header {
    background: linear-gradient(160deg, #0e0e0e, #0a0a0a);
    border: 1px solid #181818; border-left: 3px solid #c9a84c44;
    border-radius: 10px; padding: 1.3rem 1.6rem; margin-bottom: 1.8rem;
}
.panel-title { font-family: 'Bebas Neue', cursive; font-size: 1.9rem; letter-spacing: 0.15em; color: #c9a84c; margin-bottom: 0.25rem; }
.panel-desc  { font-family: 'Playfair Display', serif; font-style: italic; font-size: 0.83rem; color: #3a3a3a; }

/* Input group wrapper */
.input-group {
    background: #0c0c0c; border: 1px solid #181818; border-radius: 8px;
    padding: 1rem 1.2rem 0.8rem; margin-bottom: 1rem;
}

.lbl { font-size: 0.6rem; letter-spacing: 0.22em; text-transform: uppercase; color: #484848; font-family: 'DM Sans', sans-serif; margin-bottom: 0.2rem; margin-top: 0.6rem; display: block; }

/* Inputs */
.stTextInput input, .stNumberInput input {
    background: #0a0a0a !important; border: 1px solid #1e1e1e !important;
    border-radius: 6px !important; color: #e8e0d0 !important;
    font-family: 'DM Sans', sans-serif !important; font-size: 0.88rem !important;
    padding: 0.5rem 0.75rem !important;
}
.stTextInput input:focus { border-color: #c9a84c44 !important; box-shadow: 0 0 0 2px #c9a84c0d !important; }
.stTextInput input::placeholder { color: #282828 !important; font-style: italic !important; }
div[data-baseweb="select"] > div { background: #0a0a0a !important; border-color: #1e1e1e !important; color: #e8e0d0 !important; border-radius: 6px !important; }
div[data-baseweb="select"] li { background: #0f0f0f !important; color: #c8c0b0 !important; }

/* Buttons */
.stButton button {
    background: linear-gradient(135deg, #181200, #241a00) !important;
    border: 1px solid #c9a84c44 !important; color: #c9a84c !important;
    font-family: 'DM Sans', sans-serif !important; font-size: 0.7rem !important;
    letter-spacing: 0.18em !important; text-transform: uppercase !important;
    border-radius: 6px !important; transition: all 0.25s !important; font-weight: 500 !important;
    width: 100% !important; padding: 0.55rem !important;
}
.stButton button:hover { border-color: #c9a84c88 !important; box-shadow: 0 0 18px #c9a84c1a !important; background: linear-gradient(135deg, #1e1600, #2e2200) !important; }
.stFormSubmitButton button {
    background: linear-gradient(135deg, #181200, #241a00) !important;
    border: 1px solid #c9a84c44 !important; color: #c9a84c !important;
    font-family: 'DM Sans', sans-serif !important; font-size: 0.7rem !important;
    letter-spacing: 0.18em !important; text-transform: uppercase !important;
    border-radius: 6px !important; transition: all 0.25s !important; padding: 0.55rem !important;
}
.stFormSubmitButton button:hover { border-color: #c9a84c88 !important; box-shadow: 0 0 18px #c9a84c1a !important; }

/* Chat bubbles */
.msg-user { display: flex; justify-content: flex-end; margin: 1rem 0 0.3rem 0; }
.msg-user .bubble {
    background: linear-gradient(135deg, #181200, #241a00); border: 1px solid #c9a84c33;
    border-radius: 16px 16px 2px 16px; padding: 0.6rem 1rem; max-width: 55%;
    font-size: 0.85rem; color: #d4b060; letter-spacing: 0.01em;
}
.msg-bot { display: flex; justify-content: flex-start; margin: 0.3rem 0 0.5rem 0; align-items: flex-start; }
.msg-bot .bubble {
    background: #0f0f0f; border: 1px solid #1a1a1a; border-radius: 2px 16px 16px 16px;
    padding: 0.7rem 1.1rem; max-width: 94%; font-size: 0.85rem; color: #b8b0a0; line-height: 1.7;
}
.msg-avatar {
    width: 30px; height: 30px; border-radius: 50%;
    background: linear-gradient(135deg, #141000, #c9a84c1a); border: 1px solid #c9a84c33;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.78rem; margin-right: 0.6rem; flex-shrink: 0; margin-top: 3px;
}

/* Result pair separator */
.result-pair { border-bottom: 1px solid #0f0f0f; padding-bottom: 1.2rem; margin-bottom: 0.8rem; }

.tool-badge {
    display: inline-block; font-size: 0.56rem; font-weight: 600;
    letter-spacing: 0.22em; text-transform: uppercase; padding: 2px 9px;
    border-radius: 20px; margin-bottom: 0.5rem; font-family: 'DM Sans', sans-serif;
}
.badge-vibe    { background:#0a0a1c; color:#6e8ef5; border:1px solid #22225a; }
.badge-mood    { background:#0a1c0a; color:#6ec890; border:1px solid #1a441a; }
.badge-compare { background:#1c0a0a; color:#f09070; border:1px solid #441a1a; }
.badge-gems    { background:#1c1c0a; color:#d8c870; border:1px solid #44441a; }
.badge-llm     { background:#0a1c1c; color:#70c8d8; border:1px solid #1a4444; }

/* Movie cards */
.movies-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 1rem; margin-top: 1rem; }
.movie-card {
    background: linear-gradient(170deg, #131313, #0e0e0e); border: 1px solid #1c1c1c;
    border-radius: 10px; padding: 1.1rem 1.2rem; transition: all 0.3s; position: relative;
    overflow: hidden; cursor: default;
}
.movie-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, #c9a84c22, transparent);
    transition: all 0.3s;
}
.movie-card:hover { border-color: #c9a84c44; transform: translateY(-3px); box-shadow: 0 12px 30px #00000066; }
.movie-card:hover::before { background: linear-gradient(90deg, transparent, #c9a84c55, transparent); }
.movie-director { font-size:0.67rem; color:#484848; margin-bottom:0.35rem; font-style: italic; }
.ov-short { display: block; font-size:0.75rem; color:#666; line-height:1.65; font-style:italic; }
.ov-full  { display: none;  font-size:0.75rem; color:#908880; line-height:1.75; font-style:italic; }
.movie-card:hover .ov-short { display: none; }
.movie-card:hover .ov-full  { display: block; }
.movie-title   { font-family:'Playfair Display',serif; font-size:1.05rem; font-weight:700; color:#f0e8d8; margin-bottom:0.1rem; line-height:1.3; }
.movie-meta    { font-size:0.65rem; color:#484848; letter-spacing:0.12em; text-transform:uppercase; margin-bottom:0.5rem; }
.movie-rating-high { display:inline-block; background:linear-gradient(135deg,#0a1a0a,#142014); border:1px solid #2a6a2a44; color:#7aba7a; font-size:0.65rem; padding:2px 7px; border-radius:10px; margin-bottom:0.45rem; font-weight:500; }
.movie-rating-mid  { display:inline-block; background:linear-gradient(135deg,#1a1400,#241c00); border:1px solid #c9a84c33; color:#c9a84c; font-size:0.65rem; padding:2px 7px; border-radius:10px; margin-bottom:0.45rem; font-weight:500; }
.movie-rating-low  { display:inline-block; background:linear-gradient(135deg,#1a0a0a,#240e0e); border:1px solid #884a2a33; color:#a06040; font-size:0.65rem; padding:2px 7px; border-radius:10px; margin-bottom:0.45rem; font-weight:500; }
.movie-overview{ font-size:0.75rem; color:#666; line-height:1.65; font-style:italic; }
.movie-genres  { margin-top:0.6rem; display:flex; flex-wrap:wrap; gap:4px; }
.genre-pill    { font-size:0.56rem; letter-spacing:0.08em; text-transform:uppercase; padding:2px 6px; border-radius:8px; background:#141414; color:#3a3a3a; border:1px solid #1e1e1e; }

/* Compare */
.compare-wrap { display:grid; grid-template-columns:1fr 44px 1fr; gap:1.2rem; align-items:start; margin-top:1rem; }
.vs-div  { font-family:'Bebas Neue',cursive; font-size:2rem; color:#c9a84c88; text-align:center; padding-top:2rem; line-height:1; }
.vs-line { width:1px; height:100%; background: linear-gradient(180deg, transparent, #c9a84c22, transparent); margin: 0 auto; }
.cmp-card{ background:#0f0f0f; border:1px solid #1a1a1a; border-radius:10px; padding:1.2rem; }
.cmp-title{ font-family:'Playfair Display',serif; font-size:1.1rem; font-weight:700; color:#f0e8d8; margin-bottom:0.8rem; padding-bottom:0.5rem; border-bottom:1px solid #181818; }
.cmp-row { display:flex; justify-content:space-between; align-items:center; padding:0.3rem 0; font-size:0.78rem; border-bottom:1px solid #0d0d0d; }
.cmp-lbl { color:#383838; font-size:0.62rem; letter-spacing:0.12em; text-transform:uppercase; }
.cmp-val { color:#b0a898; font-size:0.8rem; }
.cmp-overview { margin-top:0.8rem; font-size:0.73rem; color:#484848; font-style:italic; line-height:1.65; border-top:1px solid #131313; padding-top:0.7rem; }
.verdict-box { margin-top:1rem; background:#0a0a0a; border:1px solid #181818; border-left:2px solid #c9a84c55; border-radius:8px; padding:1rem 1.2rem; font-size:0.82rem; color:#908070; line-height:1.75; font-style:italic; }

.empty-state { text-align:center; padding:4rem 2rem; }
.empty-big   { font-family:'Bebas Neue',cursive; font-size:2.5rem; letter-spacing:0.25em; color:#141414; margin-bottom:0.5rem; }
.empty-small { font-family:'Playfair Display',serif; font-style:italic; font-size:0.85rem; color:#1e1e1e; }

/* Sidebar */
section[data-testid="stSidebar"] { background:#050505 !important; border-right:1px solid #0e0e0e; }
.sb-logo { font-family:'Bebas Neue',cursive; font-size:1.6rem; letter-spacing:0.22em; color:#c9a84c; }
.sb-sub  { font-family:'Playfair Display',serif; font-style:italic; font-size:0.72rem; color:#242424; }
.sb-sec  { font-size:0.57rem; letter-spacing:0.22em; text-transform:uppercase; color:#242424; margin:1.2rem 0 0.4rem 0; }
.sb-tool-pill { display:inline-block; background:#0f0f0f; border:1px solid #1a1a1a; border-radius:20px; padding:3px 10px; font-size:0.7rem; color:#444; margin-top:0.3rem; }

@keyframes fadeIn { from{opacity:0;transform:translateY(8px)} to{opacity:1;transform:translateY(0)} }
@keyframes fadeInFast { from{opacity:0} to{opacity:1} }
.movie-card { animation:fadeIn 0.35s ease; }
.msg-bot,.msg-user { animation:fadeInFast 0.2s ease; }
.stSpinner > div { border-top-color:#c9a84c !important; }
[data-testid="stSlider"] > div > div > div { background: #c9a84c !important; }
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
        r = requests.post(f"{API_URL}/{endpoint}", json=payload, timeout=60)
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
def render_cards(movies):
    if not movies: return
    cards = ""
    for m in movies:
        title    = m.get("title",        "Unknown")
        year     = int(m.get("year", 0)) if m.get("year") else "—"
        rating   = m.get("vote_average", 0) or 0
        overview = m.get("overview",     "No synopsis on record.")
        genres   = m.get("genres",       "")
        director = m.get("director",     "")
        # Color-coded rating
        if rating >= 7.5:
            rating_cls = "movie-rating-high"
        elif rating >= 6.0:
            rating_cls = "movie-rating-mid"
        else:
            rating_cls = "movie-rating-low"
        rd       = f"★ {rating:.1f}" if rating else "Unrated"
        pills    = "".join(f'<span class="genre-pill">{g.strip()}</span>' for g in str(genres).split(",")[:3] if g.strip())
        dir_html = ""
        short_ov = overview[:140] + "..." if len(overview) > 140 else overview
        cards += (
            f'<div class="movie-card">' +
            f'<div class="movie-title">{title}</div>' +
            f'<div class="movie-meta">{year}</div>' +
            f'{dir_html}' +
            f'<div class="{rating_cls}">{rd}</div>' +
            f'<div class="movie-overview">' +
            f'  <span class="ov-short">{short_ov}</span>' +
            f'  <span class="ov-full">{overview}</span>' +
            f'</div>' +
            f'<div class="movie-genres">{pills}</div>' +
            f'</div>'
        )
    st.markdown(f'<div class="movies-grid">{cards}</div>', unsafe_allow_html=True)


def render_compare(data):
    if not data or not isinstance(data, dict):
        st.markdown(f'<div style="color:#888;font-style:italic;">{data}</div>', unsafe_allow_html=True)
        return

    def card(m, color, tone=None, watch_if=None):
        rating = m.get("vote_average", 0) or 0
        director = m.get("director", "") or ""
        fields = [
            ("Year",     m.get("year", "—")),
            ("Rating",   f"★ {rating:.1f} / 10" if rating else "—"),
            ("Votes",    f"{int(m.get('vote_count', 0)):,}" if m.get("vote_count") else "—"),
            ("Genres",   m.get("genres", "—")),
        ]
        if director:
            fields.append(("Director", director))
        if tone:
            fields.append(("Tone", tone))
        rows = "".join(f'<div class="cmp-row"><span class="cmp-lbl">{k}</span><span class="cmp-val">{v}</span></div>' for k,v in fields)
        ov = m.get("overview", "")
        ov_html = f'<div class="cmp-overview">{ov}</div>' if ov else ""
        watch_html = f'<div style="margin-top:0.7rem;font-size:0.72rem;color:#555;font-style:italic;border-top:1px solid #131313;padding-top:0.6rem;">Watch if: {watch_if}</div>' if watch_if else ""
        return f'<div class="cmp-card" style="border-top:2px solid {color}33;"><div class="cmp-title">{m.get("title","?")}</div>{rows}{ov_html}{watch_html}</div>'

    tone1    = data.get("tone_movie1", "")
    tone2    = data.get("tone_movie2", "")
    watch1   = data.get("watch_movie1_if", "")
    watch2   = data.get("watch_movie2_if", "")

    # Similarity score
    sim = data.get("tfidf_similarity")
    sim_html = ""
    if sim is not None:
        pct = int(sim * 100)
        sim_html = f'<div style="text-align:center;font-size:0.62rem;color:#333;letter-spacing:0.15em;text-transform:uppercase;margin-bottom:0.5rem;">{pct}% similar</div>'

    # Shared themes
    themes = data.get("shared_themes", [])
    themes_html = ""
    if themes:
        pills = "".join(f'<span style="background:#141414;border:1px solid #1e1e1e;border-radius:8px;font-size:0.58rem;letter-spacing:0.08em;text-transform:uppercase;padding:2px 7px;color:#444;margin-right:4px;">{t}</span>' for t in themes[:4])
        themes_html = f'<div style="margin:0.6rem 0 0.8rem 0;">Shared themes: {pills}</div>'

    # Similarity summary
    summary = data.get("similarity_summary", "")
    summary_html = f'<div style="font-size:0.78rem;color:#555;font-style:italic;margin-bottom:0.8rem;line-height:1.6;">{summary}</div>' if summary else ""

    st.markdown(
        sim_html + summary_html + themes_html +
        f'<div class="compare-wrap">{card(data.get("movie1",{}), "#7a9fff", tone1, watch1)}<div class="vs-div">VS</div>{card(data.get("movie2",{}), "#ff9f7f", tone2, watch2)}</div>',
        unsafe_allow_html=True
    )

    verdict = data.get("verdict") or data.get("analysis") or data.get("summary")
    if verdict:
        st.markdown(f'<div class="verdict-box">🎬 {verdict}</div>', unsafe_allow_html=True)

def render_bot(tool, results, raw={}):
    badges = {"vibe":"badge-vibe","mood":"badge-mood","compare":"badge-compare","gems":"badge-gems","llm":"badge-llm"}
    labels = {"vibe":"Similar Vibes","mood":"Mood Match","compare":"Face-Off","gems":"Hidden Gems","llm":"Oracle Says"}
    badge  = f'<div class="tool-badge {badges.get(tool,"badge-llm")}">{labels.get(tool,tool.upper())}</div>'
    intro  = TOOL_INTROS.get(tool, "🎬 The oracle has spoken...")

    if tool == "compare":
        st.markdown(f'<div class="msg-bot"><div class="msg-avatar">🎬</div><div class="bubble">{badge}<div style="font-style:italic;color:#555;font-size:0.76rem;">{intro}</div></div></div>', unsafe_allow_html=True)
        if results and isinstance(results, dict):
            render_compare(results)
        else:
            err = raw.get("error") or raw.get("detail") or "Could not compare those films — try adding a year to disambiguate."
            st.markdown(f'<div style="color:#c06060;font-style:italic;padding:0.5rem;">{err}</div>', unsafe_allow_html=True)
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

    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown('<span class="lbl">Movie Title — any language, any era</span>', unsafe_allow_html=True)
        v_title = st.text_input("##vt", placeholder="e.g. Inception, 3 Idiots, Parasite, Amélie...", label_visibility="collapsed")
    with c2:
        st.markdown('<span class="lbl">Release Year (if ambiguous)</span>', unsafe_allow_html=True)
        v_year = st.number_input("##vy", min_value=1900, max_value=2027, value=None, placeholder="e.g. 2010", label_visibility="collapsed", format="%d")

    c3, c4 = st.columns([2, 1])
    with c3:
        st.markdown('<span class="lbl">Era Filter (optional)</span>', unsafe_allow_html=True)
        v_era = st.selectbox("##ve", ERA_OPTIONS, label_visibility="collapsed")
    with c4:
        st.markdown('<span class="lbl">Results</span>', unsafe_allow_html=True)
        v_topn = st.selectbox("##vn", [5, 8, 10, 15], label_visibility="collapsed")

    c_btn1, c_btn2 = st.columns([3, 1])
    with c_btn1:
        v_go = st.button("✶ Find My Soulmates", key="vibe_go", use_container_width=True)
    with c_btn2:
        v_more_btn = st.button("↺ Show More", key="vibe_more", use_container_width=True)

    if v_go or v_more_btn:
        if v_more_btn and not st.session_state.get("last_vibe_title"):
            st.warning("Search for a film first, then use Show More.")
        elif not v_title.strip() and not v_more_btn:
            st.warning("Name a film first. The oracle needs something to work with.")
        else:
            with st.spinner("Searching the cinematic multiverse..."):
                if v_more_btn:
                    payload = {"query_title": "more", "top_n": v_topn, "refine": "more"}
                else:
                    payload = {"query_title": v_title.strip(), "top_n": v_topn}
                    if v_year: payload["year"] = int(v_year)
                    if v_era != "Any Era": payload["era"] = v_era.lower()
                resp = api("recommend", payload)

            if resp.get("status") == "disambig":
                options = resp.get("options", [])
                st.warning(f"Multiple films named **{v_title.strip()}** found. Enter the year to disambiguate:")
                for opt in options:
                    yr  = int(opt.get("year", 0))
                    ttl = opt.get("title", "")
                    gen = opt.get("genres", "")
                    rat = opt.get("vote_average", "")
                    st.markdown(f'<div style="background:#0f0f0f;border:1px solid #1e1e1e;border-radius:6px;padding:0.5rem 0.8rem;margin-bottom:0.3rem;font-size:0.82rem;color:#888;"><span style="color:#c9a84c;font-weight:600;">{yr}</span> — {ttl} <span style="color:#444;font-size:0.72rem;">({gen})</span> ★ {rat}</div>', unsafe_allow_html=True)
                st.info("Enter the release year in the Release Year field above and search again.")
            else:
                if v_more_btn:
                    label = f'↺ More like "{st.session_state.get("last_vibe_title", "?")}"' + (f" · {v_era}" if v_era != "Any Era" else "")
                else:
                    label = f'Films like "{v_title.strip()}"' + (f" · {v_era}" if v_era != "Any Era" else "")
                    st.session_state["last_vibe_title"] = v_title.strip()
                    if v_era != "Any Era": st.session_state["last_vibe_era"] = v_era.lower()
                    else: st.session_state.pop("last_vibe_era", None)
                push(label, "vibe", resp.get("results"), resp)
                st.rerun()

# ── TAB 2: MOOD ENGINE ────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="panel-header"><div class="panel-title">🌙 MOOD ENGINE</div><div class="panel-desc">"Tell us how you feel. We\'ll find exactly what you need to watch."</div></div>', unsafe_allow_html=True)
    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown('<span class="lbl">Your Mood / Feeling / Occasion / Craving</span>', unsafe_allow_html=True)
        st.markdown('<div style="font-size:0.72rem;color:#444;font-style:italic;margin-bottom:0.3rem;">💡 You can mention a language too — e.g. "sad and nostalgic, Hindi" or "feel-good Korean"</div>', unsafe_allow_html=True)
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
                if resp.get("status") == "disambig":
                    which    = resp.get("which", "title1")
                    options  = resp.get("options", [])
                    title_lbl = c_t1.strip() if which == "title1" else c_t2.strip()
                    year_lbl  = "Release Year 1" if which == "title1" else "Release Year 2"
                    st.warning(f"Multiple films named **{title_lbl}** found. Enter the year to disambiguate:")
                    for opt in options:
                        yr  = int(opt.get("year", 0)) if opt.get("year") else "?"
                        ttl = opt.get("title", "")
                        gen = opt.get("genres", "")
                        rat = opt.get("vote_average", "")
                        st.markdown(f'<div style="background:#0f0f0f;border:1px solid #1e1e1e;border-radius:6px;padding:0.5rem 0.8rem;margin-bottom:0.3rem;font-size:0.82rem;color:#888;"><span style="color:#c9a84c;font-weight:600;">{yr}</span> — {ttl} <span style="color:#444;font-size:0.72rem;">({gen})</span> ★ {rat}</div>', unsafe_allow_html=True)
                    st.info(f"Enter the release year in the **{year_lbl}** field above and search again.")
                elif resp.get("status") == "error" and not resp.get("results"):
                    err = resp.get("error", "Could not compare those films.")
                    st.warning(f"⚠️ {err}")
                else:
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

    g_col1, g_col2 = st.columns([2, 1])
    with g_col1:
        g_go_btn   = st.button("✦ Hunt the Gems", key="gems_go",   use_container_width=True)
    with g_col2:
        g_more_btn = st.button("↺ Show More",     key="gems_more", use_container_width=True)

    if g_go_btn or g_more_btn:
        with st.spinner("Digging through the vaults of forgotten cinema..."):
            prev_genre = st.session_state.get("gems_last_genre")
            prev_era   = st.session_state.get("gems_last_era")
            if g_go_btn and (prev_genre != g_genre or prev_era != g_era):
                st.session_state["gems_shown"] = []
            st.session_state["gems_last_genre"] = g_genre
            st.session_state["gems_last_era"]   = g_era

            shown   = st.session_state.get("gems_shown", [])
            payload = {"top_n": g_topn, "min_votes": g_votes, "max_popularity": g_pop,
                       "exclude_titles": shown}
            if g_more_btn:
                payload["refine"] = "more"
                if prev_genre and prev_genre != "Any Genre": payload["genre"] = prev_genre
                if prev_era   and prev_era   != "Any Era":   payload["era"]   = prev_era.lower()
            else:
                if g_genre != "Any Genre": payload["genre"] = g_genre
                if g_era   != "Any Era":   payload["era"]   = g_era.lower()

            resp   = api("gems", payload)
            titles = [r.get("title", "") for r in (resp.get("results") or [])]
            st.session_state["gems_shown"] = list(dict.fromkeys(shown + titles))

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
    st.markdown('<div style="font-size:0.55rem;letter-spacing:0.25em;text-transform:uppercase;color:#1e1e1e;margin-bottom:1rem;">— Latest Results First —</div>', unsafe_allow_html=True)
    # Pair messages as (user, assistant) and show newest pair first
    msgs = st.session_state.messages
    pairs = [(msgs[i], msgs[i+1]) for i in range(0, len(msgs)-1, 2) if i+1 < len(msgs)]
    for user_msg, bot_msg in reversed(pairs):
        st.markdown('<div class="result-pair">', unsafe_allow_html=True)
        st.markdown(f'<div class="msg-user"><div class="bubble">{user_msg.get("content","")}</div></div>', unsafe_allow_html=True)
        render_bot(bot_msg.get("tool","llm"), bot_msg.get("results"), bot_msg.get("raw",{}))
        st.markdown('</div>', unsafe_allow_html=True)