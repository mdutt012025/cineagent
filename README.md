# 🎬 CineAgent

A multi-tool AI movie recommendation engine powered by TF-IDF similarity search, FAISS vector embeddings, and a Groq LLM agent that routes every query to the right tool automatically.

---

## How It Works

Every message goes through a **Groq-powered routing agent** that reads the user's intent and dispatches to one of five specialised tools:

| Tool | Triggered by | Example |
|------|-------------|---------|
| **Vibe** | "something like [movie]" | *"Movies like Interstellar"* |
| **Mood** | feeling / occasion / genre | *"Something for a rainy Sunday"* |
| **Compare** | vs / compare / better than | *"Alien vs The Thing"* |
| **Gems** | hidden gem / underrated | *"Underrated 90s horror"* |
| **LLM Chat** | everything else | *"Best Nolan films ranked"* |

---

## Architecture

```
User message
    │
    ▼
[Agent] groq_route_message()        ← LLM decides which tool to call
    │
    ├── [Vibe]    recommend_by_vibe()
    │       ├── TF-IDF / FAISS candidate pool
    │       ├── Emotional axis scoring (tempo, darkness, warmth…)
    │       ├── MMR diversity filter
    │       └── Groq LLM ranker + explainer
    │
    ├── [Mood]    mood_based_recommend()
    │       ├── Mood → genre mapping (joblib model)
    │       └── Era + language filtering
    │
    ├── [Compare] compare_movies()
    │       └── Side-by-side Groq LLM analysis
    │
    ├── [Gems]    discover_hidden_gems()
    │       └── Weighted hidden gem scoring (quality / popularity ratio)
    │
    └── [LLM]     llm_movie_chat()
            └── Multi-turn Groq conversation with TMDB context injection
```

---

## Tech Stack

- **Backend** — FastAPI + Uvicorn
- **Frontend** — Streamlit
- **LLM** — Groq (Llama 3.3 70B)
- **Similarity search** — TF-IDF (sklearn) + FAISS (sentence-transformers)
- **Dataset** — TMDB 90k movies (parquet)
- **Live data** — TMDB API (posters, recent films, ratings enrichment)

---

## Project Structure

```
cineagent/
├── backend/
│   ├── main.py                    # FastAPI app, all endpoints
│   ├── requirements.txt
│   ├── Dockerfile
│   └── tools/
│       ├── agent.py               # LLM routing agent
│       ├── vibe.py                # Similarity recommendation engine
│       ├── mood.py                # Mood-based recommendations
│       ├── compare.py             # Movie comparison tool
│       ├── gems.py                # Hidden gems discovery
│       ├── llm_chat.py            # Multi-turn LLM chatbot
│       ├── shared.py              # Shared data, TMDB client, artifacts loader
│       ├── generate_embeddings.py # One-time: build sentence-transformer embeddings
│       ├── build_faiss_index.py   # One-time: build FAISS index from embeddings
│       └── rebuild_soup.py        # Run after dataset changes: rebuild TF-IDF matrix
├── frontend/
│   ├── app.py                     # Streamlit UI
│   └── requirements.txt
├── docker-compose.yml
└── .gitignore
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/vibe` | Main agent — routes any message automatically |
| POST | `/recommend` | Direct vibe/similarity search |
| POST | `/mood` | Mood-based recommendations |
| POST | `/compare` | Compare two movies |
| POST | `/gems` | Discover hidden gems |
| POST | `/llm_chat` | Multi-turn movie chatbot |
| GET | `/poster` | Fetch TMDB poster URL |

Interactive docs available at `http://localhost:8000/docs` when the backend is running.