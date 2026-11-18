#!/usr/bin/env bash
set -e
TARGET_DIR="/kaggle/working/Multi-Agent_RAG"

if [ ! -d "$TARGET_DIR" ]; then
    git clone -q https://github.com/ducmanh-jr/Multi-Agent_RAG.git "$TARGET_DIR" 2>/dev/null
else
    git -C "$TARGET_DIR" fetch -q origin && git -C "$TARGET_DIR" reset --hard -q origin/main
fi

cd "$TARGET_DIR"
python3 -u kaggle/run_kaggle.py
