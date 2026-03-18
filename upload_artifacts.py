"""
Run ONCE from your local machine.
Uploads model_artifacts/ to a private Hugging Face dataset repo.

    pip install huggingface_hub
    python upload_artifacts.py
"""
from pathlib import Path
from huggingface_hub import HfApi, create_repo

HF_TOKEN    = input("HF token (hf.co/settings/tokens → New → Write access): ").strip()
HF_USERNAME = input("Your HF username: ").strip()
REPO_NAME   = "cineagent-artifacts"
ARTIFACTS   = Path(__file__).parent / "backend" / "model_artifacts"

REPO_ID = f"{HF_USERNAME}/{REPO_NAME}"
api = HfApi(token=HF_TOKEN)

print(f"\nCreating private dataset repo: {REPO_ID} ...")
try:
    create_repo(REPO_ID, repo_type="dataset", private=True, token=HF_TOKEN)
    print("Repo created.")
except Exception as e:
    print(f"Already exists or error: {e}")

FILES = [
    "movies_enriched.parquet",
    "tfidf_vectorizer.joblib",
    "tfidf_matrix.joblib",
    "title_to_idx.joblib",
]

for fname in FILES:
    p = ARTIFACTS / fname
    if not p.exists():
        print(f"  SKIP — not found: {p}")
        continue
    mb = p.stat().st_size / 1024 / 1024
    print(f"  Uploading {fname}  ({mb:.0f} MB)...")
    api.upload_file(
        path_or_fileobj=str(p),
        path_in_repo=fname,
        repo_id=REPO_ID,
        repo_type="dataset",
        token=HF_TOKEN,
    )
    print(f"  ✓ done")

print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Upload complete!

Add these as Secrets in your HF Space:
  HF_REPO  = {REPO_ID}
  HF_TOKEN = {HF_TOKEN[:12]}...
  GROQ_API_KEY = your_groq_key
  TMDB_API_KEY = your_tmdb_key
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
