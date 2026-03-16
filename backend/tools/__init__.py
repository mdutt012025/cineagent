# backend/tools/__init__.py
# exposes all tools and agent cleanly for main.py to import

from .vibe      import recommend_by_vibe
from .mood      import mood_based_recommend
from .compare   import compare_movies
from .gems      import discover_hidden_gems
from .llm_chat  import llm_movie_chat, get_chat_history, clear_chat_history
from .agent     import run_agent