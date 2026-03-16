"""
rebuild_soup.py
Run whenever you update the dataset or change the soup formula.
Rebuilds tfidf_matrix.joblib and title_to_idx.joblib.

Usage:
    cd C:/Users/91981/cineagent
    python rebuild_soup.py
"""

import pandas as pd
import joblib
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR  = Path(__file__).resolve().parent.parent.parent  # tools → backend → cineagent
ARTIFACTS = BASE_DIR / 'model_artifacts'

LANG_MAP = {
    'hi': 'hindi bollywood indian',
    'ta': 'tamil kollywood indian',
    'te': 'telugu tollywood indian',
    'ml': 'malayalam indian',
    'kn': 'kannada indian',
    'ko': 'korean',
    'ja': 'japanese',
    'fr': 'french',
    'es': 'spanish',
    'de': 'german',
    'it': 'italian',
    'zh': 'chinese mandarin',
    'pt': 'portuguese',
    'ru': 'russian',
    'ar': 'arabic',
    'tr': 'turkish',
    'mr': 'marathi indian',
    'pa': 'punjabi indian',
    'en': 'english hollywood',
}

def build_soup(row):
    parts = []
    parts.extend([str(row.get('genre_str',   ''))] * 3)
    parts.extend([str(row.get('keyword_str', ''))] * 6)
    lang = str(row.get('original_language', 'en')).strip().lower()
    parts.extend([LANG_MAP.get(lang, lang)] * 4)
    parts.extend([str(row.get('overview', ''))] * 4)
    tagline = str(row.get('tagline', ''))
    if tagline and tagline != 'nan':
        parts.extend([tagline] * 2)
    return ' '.join(p for p in parts if p and p != 'nan')

print('[soup] Loading dataset...')
parquet_path = ARTIFACTS / 'movies_enriched.parquet'
if not parquet_path.exists():
    print(f'[soup] ERROR: {parquet_path} not found.')
    print(f'[soup] Expected location: {BASE_DIR}')
    exit(1)

df = pd.read_parquet(str(parquet_path))
print(f'[soup] {len(df):,} movies loaded')

print('[soup] Rebuilding content soups (keyword x6)...')
df['content_soup'] = df.apply(build_soup, axis=1)

vectorizer_path = ARTIFACTS / 'tfidf_vectorizer.joblib'
if not vectorizer_path.exists():
    print(f'[soup] ERROR: tfidf_vectorizer.joblib not found at {vectorizer_path}')
    exit(1)

tfidf = joblib.load(str(vectorizer_path))
print('[soup] Rebuilding TF-IDF matrix...')
matrix = tfidf.transform(df['content_soup'].fillna(''))

print('[soup] Saving...')
df.to_parquet(str(parquet_path), index=False)
joblib.dump(matrix, str(ARTIFACTS / 'tfidf_matrix.joblib'))

title_to_idx = pd.Series(df.index, index=df['title_clean']).to_dict()
joblib.dump(title_to_idx, str(ARTIFACTS / 'title_to_idx.joblib'))

print('[soup] Done! Running verification...')
print(f'  df rows    : {len(df):,}')
print(f'  matrix rows: {matrix.shape[0]:,}')
print(f'  In sync    : {len(df) == matrix.shape[0]}')

for test_title in ['yeh jawaani hai deewani', 'parasite']:
    if test_title in title_to_idx:
        row = df.iloc[title_to_idx[test_title]]
        print(f'  [{test_title}] soup: {row["content_soup"][:200]}')

test_title = 'yeh jawaani hai deewani'
if test_title in title_to_idx:
    pos = title_to_idx[test_title]
    sim = cosine_similarity(matrix[pos], matrix).flatten()
    neighbours = ['zindagi na milegi dobara', 'dil chahta hai', 'jab we met',
                  'kal ho naa ho', 'tamasha', 'wake up sid']
    for t in neighbours:
        if t in title_to_idx:
            rank = int((sim > sim[title_to_idx[t]]).sum())
            print(f'  {t}: sim={sim[title_to_idx[t]]:.3f} rank={rank}')
else:
    print(f'[soup] Skipping rank check — "{test_title}" not in dataset')

print('[soup] Next steps:')
print('  python generate_embeddings.py   (if not already done)')
print('  python build_faiss_index.py     (if not already done)')
print('  uvicorn backend.main:app --workers 1')