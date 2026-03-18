from huggingface_hub import HfApi
from pathlib import Path

TOKEN    = input("Your HF token: ").strip()
BACKEND  = Path("C:/Users/91981/cineagent/backend")
REPO_ID  = "mahiii2341/cineagent"

api = HfApi(token=TOKEN)

# Files to upload
files = [
    "Dockerfile",
    "main.py",
    "requirements.txt",
    "start.sh",
    "tools/__init__.py",
    "tools/agent.py",
    "tools/vibe.py",
    "tools/shared.py",
    "tools/mood.py",
    "tools/compare.py",
    "tools/gems.py",
    "tools/llm_chat.py",
    "tools/build_faiss_index.py",
    "tools/generate_embeddings.py",
    "tools/rebuild_soup.py",
]

for f in files:
    path = BACKEND / f
    if not path.exists():
        print(f"  SKIP (not found): {f}")
        continue
    print(f"  Uploading {f}...")
    api.upload_file(
        path_or_fileobj=str(path),
        path_in_repo=f,
        repo_id=REPO_ID,
        repo_type="space",
        token=TOKEN,
    )
    print(f"  done")

print("\nAll files uploaded! Check your Space now.")
