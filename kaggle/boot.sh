#!/usr/bin/env bash
set -Eeuo pipefail

REPO_URL="${REPO_URL:-https://github.com/ducmanh-jr/Multi-Agent_RAG.git}"
BRANCH="${BRANCH:-tatmanh}"
GIT_COMMIT_SHA="${GIT_COMMIT_SHA:-}"
REQUIRE_PINNED_COMMIT="${REQUIRE_PINNED_COMMIT:-false}"
KAGGLE_WORKER_PROFILE="${KAGGLE_WORKER_PROFILE:-qa_dual_gpu}"
TARGET_DIR="${TARGET_DIR:-/kaggle/working/Multi-Agent_RAG}"

is_truthy() {
    case "$(printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]')" in
        true|1|yes) return 0 ;;
        *) return 1 ;;
    esac
}

validate_sha40() {
    printf '%s' "${1:-}" | grep -Eq '^[0-9a-fA-F]{40}$'
}

if [ -n "$GIT_COMMIT_SHA" ]; then
    if ! validate_sha40 "$GIT_COMMIT_SHA"; then
        echo "[FATAL] GIT_COMMIT_SHA must be a full 40-character hexadecimal SHA, got: '$GIT_COMMIT_SHA'"
        exit 1
    fi
elif is_truthy "$REQUIRE_PINNED_COMMIT"; then
    echo "[FATAL] REQUIRE_PINNED_COMMIT=true requires a pinned 40-char GIT_COMMIT_SHA!"
    exit 1
fi

echo "=========================================================================="
echo "🚀 [KAGGLE RUNTIME BOOTSTRAPPER] Reproducible Pinned Multi-Profile Runtime"
echo "📌 Worker Profile: $KAGGLE_WORKER_PROFILE"
if [ -n "$GIT_COMMIT_SHA" ]; then
    echo "🔒 Pinned Commit SHA: $GIT_COMMIT_SHA"
else
    echo "📌 Target Branch (unpinned): $BRANCH"
fi
echo "📂 Target Directory: $TARGET_DIR"
echo "=========================================================================="

if [ -n "$GIT_COMMIT_SHA" ]; then
    if [ ! -d "$TARGET_DIR/.git" ]; then
        if [ -e "$TARGET_DIR" ]; then
            echo "[INFO] Removing non-git directory at $TARGET_DIR..."
            rm -rf "$TARGET_DIR"
        fi
        mkdir -p "$TARGET_DIR"
        git -C "$TARGET_DIR" init -q
        git -C "$TARGET_DIR" remote add origin "$REPO_URL"
    else
        git -C "$TARGET_DIR" remote set-url origin "$REPO_URL"
    fi
    echo "[INFO] Fetching pinned commit $GIT_COMMIT_SHA..."
    if ! git -C "$TARGET_DIR" fetch -q --depth 1 origin "$GIT_COMMIT_SHA"; then
        echo "[INFO] Shallow commit fetch unsupported by remote; fetching branch $BRANCH..."
        git -C "$TARGET_DIR" fetch -q origin "$BRANCH"
    fi
    git -C "$TARGET_DIR" checkout -q --detach "$GIT_COMMIT_SHA"
    git -C "$TARGET_DIR" reset --hard -q "$GIT_COMMIT_SHA"
else
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
fi

cd "$TARGET_DIR"

ACTIVE_SHA="$(git rev-parse HEAD)"
if [ -n "$GIT_COMMIT_SHA" ]; then
    if [ "$(printf '%s' "$ACTIVE_SHA" | tr '[:upper:]' '[:lower:]')" != "$(printf '%s' "$GIT_COMMIT_SHA" | tr '[:upper:]' '[:lower:]')" ]; then
        echo "[FATAL] Checked out commit ($ACTIVE_SHA) does not match pinned GIT_COMMIT_SHA ($GIT_COMMIT_SHA)!"
        exit 1
    fi
fi

if [ ! -f "kaggle/run_kaggle.py" ]; then
    echo "[FATAL] kaggle/run_kaggle.py not found at commit $ACTIVE_SHA!"
    exit 1
fi

export ACTIVE_GIT_COMMIT_SHA="$ACTIVE_SHA"
export KAGGLE_WORKER_PROFILE

echo "[INFO] Active commit SHA: $ACTIVE_SHA"
echo "[INFO] Worker profile:    $KAGGLE_WORKER_PROFILE"
echo "=========================================================================="
python3 -u kaggle/run_kaggle.py
