# backend/main.py
# ═══════════════════════════════════════════════════════════════════════════════
# FastAPI app — exposes all 5 tools + agent as REST endpoints
# ═══════════════════════════════════════════════════════════════════════════════

import os
import traceback
import json
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / '.env')

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn

from tools import (
    recommend_by_vibe,
    mood_based_recommend,
    compare_movies,
    discover_hidden_gems,
    llm_movie_chat,
    get_chat_history,
    clear_chat_history,
    run_agent,
)

app = FastAPI(
    title       = "CineAgent API",
    description = "Movie recommendation engine powered by TF-IDF + Groq LLM",
    version     = "2.0.0",
)

# ── CORS — allows Streamlit frontend to call this API ─────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ═══════════════════════════════════════════════════════════════════════════════
# REQUEST MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class VibeRequest(BaseModel):
    query_title:    str
    top_n:          Optional[int]  = 5
    year:           Optional[int]  = None
    era:            Optional[str]  = None
    refine:         Optional[str]  = None
    exclude_titles: Optional[list] = None


class MoodRequest(BaseModel):
    mood:           str
    top_n:          Optional[int]  = 10
    era:            Optional[str]  = None
    min_votes:      Optional[int]  = 500
    refine:         Optional[str]  = None
    exclude_titles: Optional[list] = None


class CompareRequest(BaseModel):
    title1: str
    title2: str
    year1:  Optional[int] = None
    year2:  Optional[int] = None


class GemsRequest(BaseModel):
    genre:          Optional[str]   = None
    era:            Optional[str]   = None
    top_n:          Optional[int]   = 10
    min_votes:      Optional[int]   = 50
    max_popularity: Optional[float] = 30.0


class ChatRequest(BaseModel):
    message: str                        # user's natural language message


class LLMChatRequest(BaseModel):
    message: str
    reset:   Optional[bool] = False     # set True to start a fresh conversation


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
def health():
    """Health check — confirms API is running."""
    return {"status": "ok", "service": "CineAgent API", "version": "2.0.0"}


@app.post("/recommend")
def recommend(req: VibeRequest):
    """Tool 1 — recommend movies similar to a given title."""
    response = recommend_by_vibe(
        query_title    = req.query_title,
        top_n          = req.top_n,
        year           = req.year,
        era            = req.era,
        refine         = req.refine,
        exclude_titles = req.exclude_titles,
    )
    # recommend_by_vibe returns 2 or 3 values depending on disambiguation
    if len(response) == 3:
        result, err, options = response
    else:
        result, err = response
        options = None

    if err and not err.startswith('DISAMBIG'):
        return {"status": "error", "results": None, "options": None, "error": err}
    return {
        "status":  "disambig" if err and err.startswith('DISAMBIG') else "ok",
        "results": result,
        "options": options,
        "error":   err,
    }


@app.post("/mood")
def mood(req: MoodRequest):
    """Tool 2 — recommend movies based on mood, feeling, or occasion."""
    result, err = mood_based_recommend(
        mood           = req.mood,
        top_n          = req.top_n,
        era            = req.era,
        min_votes      = req.min_votes,
        refine         = req.refine,
        exclude_titles = req.exclude_titles,
    )
    # Low-confidence or niche mood — fall back to LLM
    if err == 'LLM_FALLBACK':
        result, err = llm_movie_chat(req.mood)
        if err:
            raise HTTPException(status_code=500, detail=err)
        return {"status": "ok", "tool": "llm", "results": result}
    if err:
        return {"status": "error", "results": None, "error": err}
    return {"status": "ok", "tool": "mood", "results": result}


@app.post("/compare")
def compare(req: CompareRequest):
    """Tool 3 — compare two movies side by side."""
    result, err = compare_movies(
        title1 = req.title1,
        title2 = req.title2,
        year1  = req.year1,
        year2  = req.year2,
    )
    if err:
        return {"status": "error", "results": None, "error": err}
    return {"status": "ok", "results": result}


@app.post("/gems")
def gems(req: GemsRequest):
    """Tool 4 — discover hidden gem movies."""
    result, err = discover_hidden_gems(
        genre          = req.genre,
        era            = req.era,
        top_n          = req.top_n,
        min_votes      = req.min_votes,
        max_popularity = req.max_popularity,
    )
    if err:
        return {"status": "error", "results": None, "error": err}
    return {"status": "ok", "results": result}


@app.post("/llm_chat")
def llm_chat(req: LLMChatRequest):
    """
    Tool 5 — pure LLM movie chatbot.
    Handles anything movie-related: directors, actors, year+genre combos,
    trivia, awards, complex filters, follow-up questions, etc.
    Maintains conversation history across turns.
    Set reset=True to start a fresh conversation.
    """
    result, err = llm_movie_chat(message=req.message, reset=req.reset)
    if err:
        raise HTTPException(status_code=500, detail=err)
    return {
        "status":  "ok",
        "results": result,
        "history_length": len(get_chat_history()),
    }


@app.post("/llm_chat/reset")
def llm_chat_reset():
    """Clear the LLM conversation history."""
    clear_chat_history()
    return {"status": "ok", "message": "Conversation history cleared."}


@app.get("/llm_chat/history")
def llm_chat_history():
    """Returns the current LLM conversation history."""
    return {"status": "ok", "history": get_chat_history()}


@app.post("/vibe")
def vibe_agent(req: ChatRequest):
    """
    Main agent endpoint — accepts any natural language movie message.
    Routes automatically to the best tool:
      vibe     → "something like Inception"
      mood     → "something for a rainy Sunday"
      compare  → "Alien vs Predator"
      gems     → "underrated 90s horror"
      llm      → everything else (director queries, year+genre, trivia, etc.)
    """
    result = run_agent(req.message)
    return {
        "status":  "ok",
        "tool":    result.get("tool"),
        "results": result.get("results"),
        "error":   result.get("error"),
        "message": result.get("message"),
    }


@app.get("/poster")
def get_poster(title: str, year: int = None):
    """Returns TMDB poster URL for a movie title."""
    from tools.shared import tmdb_search
    result = tmdb_search(title, year)
    if 'error' in result:
        return {"poster_url": ""}
    return {"poster_url": result.get("poster_url", "")}


# ═══════════════════════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)