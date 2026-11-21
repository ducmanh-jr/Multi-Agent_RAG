#!/usr/bin/env bash
set -Eeuo pipefail

REPO_URL="https://github.com/ducmanh-jr/Multi-Agent_RAG.git"
BRANCH="${BRANCH:-tatmanh}"
TARGET_DIR="/kaggle/working/Multi-Agent_RAG"

echo "=========================================================================="
echo "🚀 [KAGGLE RUNTIME BOOTSTRAPPER] One Runtime — Three Dynamic Modes"
echo "📌 Target Branch: $BRANCH"
echo "📂 Target Directory: $TARGET_DIR"
echo "=========================================================================="

if [ ! -d "$TARGET_DIR/.git" ]; then
    if [ -e "$TARGET_DIR" ]; then
        echo "[ERROR] Target directory exists but is not a Git repository. Removing old directory..."
        rm -rf "$TARGET_DIR"
    fi
    echo "[INFO] Cloning repository (branch: $BRANCH)..."
    git clone -q \
        --depth 1 \
        --single-branch \
        --branch "$BRANCH" \
        "$REPO_URL" \
        "$TARGET_DIR"
else
    echo "[INFO] Updating existing repository on branch $BRANCH..."
    git -C "$TARGET_DIR" fetch -q --depth 1 origin "$BRANCH"
    git -C "$TARGET_DIR" checkout -B "$BRANCH" "origin/$BRANCH"
    git -C "$TARGET_DIR" reset --hard -q "origin/$BRANCH"
fi

cd "$TARGET_DIR"

if [ ! -f "kaggle/run_kaggle.py" ]; then
    echo "[FATAL] kaggle/run_kaggle.py not found on branch $BRANCH!"
    exit 1
fi

echo "[INFO] Active branch: $BRANCH"
echo "[INFO] Active commit: $(git rev-parse --short HEAD)"
echo "=========================================================================="
python3 -u kaggle/run_kaggle.py
