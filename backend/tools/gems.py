# backend/tools/gems.py
# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 4: discover_hidden_gems
# ═══════════════════════════════════════════════════════════════════════════════

import re
import json
import pandas as pd
from groq import Groq

from .shared import (
    df, GROQ_API_KEY,
    persist_if_new, franchise_key, get_era_range,
)


def discover_hidden_gems(genre: str = None,
                         top_n: int = 10, min_votes: int = 50,
                         max_popularity: float = 30.0,
                         era: str = None):
    """
    Find highly-rated but obscure movies.
    Pipeline:
      1. Dataset filters by popularity + votes + genre + era
      2. Sorts by hidden_gem_score
      3. Groq verifies top 20 — removes mainstream, confirms genuine gems
      4. Returns Groq-verified top_n as list of dicts
    """
    _size_before = len(df)

    cands = df[
        (df['vote_count']  >= min_votes) &
        (df['popularity']  <= max_popularity)
    ].copy()

    # ── Genre filter ──────────────────────────────────────────────────────────
    if genre:
        gn = genre.lower()
        def has_genre(gl):
            return isinstance(gl, list) and any(g.lower() == gn for g in gl)
        cands = cands[cands['genre_list'].apply(has_genre)]

    # ── Taste profile filter ──────────────────────────────────────────────────


    # ── Era filter ────────────────────────────────────────────────────────────
    if era:
        year_min, year_max = get_era_range(era)
        cands_yr = pd.to_numeric(cands['year'], errors='coerce').fillna(0)
        if year_min: cands = cands[cands_yr >= year_min]
        if year_max: cands = cands[cands_yr <= year_max]
        print(f"[gems] Era filter: {era}")

    if cands.empty:
        persist_if_new(_size_before)
        return None, f"No hidden gems found" + (f" for genre '{genre}'" if genre else "")

    # ── Sort + franchise dedup ────────────────────────────────────────────────
    results        = cands.sort_values('hidden_gem_score', ascending=False).copy()
    results['_fr'] = results['title'].apply(franchise_key)
    results        = results.drop_duplicates('_fr', keep='first').drop(columns='_fr')

    top20 = results.head(max(20, top_n * 2))
    print(f"[gems] Pool: {len(cands):,} candidates → top 20 sent to Groq")

    # ── Groq verifies and reranks ─────────────────────────────────────────────
    def groq_verify_gems(candidates_df, genre_hint: str = None) -> list:
        items = []
        for i, (_, row) in enumerate(candidates_df.iterrows()):
            yr = int(row['year']) if not pd.isna(row['year']) else '?'
            items.append(
                f"{i+1}. {row['title']} ({yr}) | {row['genres']} "
                f"| rating: {row['vote_average']} "
                f"| overview: {str(row['overview'])[:100]}"
            )

        genre_context = f"Focus on {genre_hint} genre. " if genre_hint else ""
        prompt = (
            f"You are a film expert identifying genuine hidden gem movies.\n"
            f"{genre_context}"
            f"From this list of low-popularity but highly-rated films, "
            f"pick the {top_n} that are truly underrated gems — "
            f"great quality but genuinely obscure, not just temporarily forgotten blockbusters.\n"
            f"Remove anything that's actually well-known or mainstream.\n"
            f"Return ONLY a JSON array of the numbers you pick, e.g. [2,5,7,11,14].\n"
            f"No explanation.\n\n"
            + '\n'.join(items)
        )
        try:
            resp = Groq(api_key=GROQ_API_KEY).chat.completions.create(
                model='llama-3.3-70b-versatile',
                messages=[{'role': 'user', 'content': prompt}],
                temperature=0.1,
                max_tokens=100,
            )
            raw   = resp.choices[0].message.content.strip()
            raw   = re.sub(r'[^0-9,\[\]]', '', raw)
            picks = json.loads(raw)
            picks = [p - 1 for p in picks if 1 <= p <= len(candidates_df)]
            print(f"[gems] Groq verified {len(picks)} gems from 20 candidates")
            return picks
        except Exception as e:
            print(f"[gems] Groq verify failed: {e} — returning dataset order")
            return list(range(min(top_n, len(candidates_df))))

    picks = groq_verify_gems(top20, genre_hint=genre)
    final = top20.iloc[picks]

    # ── Pad if Groq returns fewer than top_n ──────────────────────────────────
    if len(final) < top_n:
        print(f"[gems] Groq returned {len(final)} — padding with dataset order")
        shown     = set(final.index)
        remaining = results[~results.index.isin(shown)].head(top_n - len(final))
        final     = pd.concat([final, remaining])

    persist_if_new(_size_before)

    out_cols = ['title', 'year', 'genres', 'vote_average',
                'vote_count', 'popularity', 'overview']
    out_cols = [c for c in out_cols if c in final.columns]
    return final.head(top_n)[out_cols].to_dict('records'), None