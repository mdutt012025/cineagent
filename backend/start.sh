#!/bin/bash
set -e

ARTIFACTS_DIR="/app/model_artifacts"
REQUIRED_FILES=(
    "movies_enriched.parquet"
    "tfidf_vectorizer.joblib"
    "tfidf_matrix.joblib"
    "title_to_idx.joblib"
)

echo "[startup] Checking artifacts at $ARTIFACTS_DIR..."
mkdir -p "$ARTIFACTS_DIR"

MISSING=0
for f in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$ARTIFACTS_DIR/$f" ]; then
        echo "[startup] Missing: $f"
        MISSING=1
    fi
done

if [ "$MISSING" = "1" ]; then
    echo "[startup] Downloading from HF dataset: $HF_REPO"
    python3 - <<EOF
import os, sys
from huggingface_hub import hf_hub_download

repo  = os.environ.get("HF_REPO", "")
token = os.environ.get("HF_TOKEN", "")

if not repo:
    print("ERROR: HF_REPO not set. Add it as a Space secret.")
    sys.exit(1)

files = [
    "movies_enriched.parquet",
    "tfidf_vectorizer.joblib",
    "tfidf_matrix.joblib",
    "title_to_idx.joblib",
]
for fname in files:
    print(f"  -> {fname}", flush=True)
    hf_hub_download(
        repo_id=repo,
        filename=fname,
        repo_type="dataset",
        local_dir="/app/model_artifacts",
        token=token or None,
    )
print("[startup] All artifacts ready.")
EOF
else
    echo "[startup] Artifacts already present."
fi

echo "[startup] Starting FastAPI on port 7860..."
exec uvicorn main:app --host 0.0.0.0 --port 7860
