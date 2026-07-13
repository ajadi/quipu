#!/bin/sh
# install-quipu-global.sh — idempotent global install for Quipu.
#
# Steps:
#   1. Resolve QUIPU_HOME (default ~/.quipu) and venv path.
#   1b. Read saved model from $QUIPU_HOME/config.
#   2. Find repo root (parent of this script's directory).
#   3. Create venv if absent.
#   4. Upgrade pip; pip install -e <repo_root>. onnxruntime wheel failure → WARN + continue.
#   5. Ensure huggingface_hub in venv (best-effort).
#   6. Prompt model picker (interactive) or use saved/default; fetch chosen model if absent.
#   7. Validate: python -m quipu --version must succeed.
#   8. Echo .mcp.json hint.
#
# Re-run is safe: no re-download if model already present.
set -eu

# Parse flags
WITH_HOOKS=0
for _arg in "$@"; do
    case "$_arg" in
        --with-hooks) WITH_HOOKS=1 ;;
    esac
done

QUIPU_HOME="${QUIPU_HOME:-$HOME/.quipu}"
VENV="$QUIPU_HOME/venv"

# Step 1b — read saved model from config
CONFIG_FILE="$QUIPU_HOME/config"
SAVED_MODEL=""
if [ -f "$CONFIG_FILE" ]; then
    SAVED_MODEL=$(grep '^MODEL=' "$CONFIG_FILE" | cut -d= -f2- | head -1)
fi

# Model picker helpers
_resolve_model() {
    # $1 = numeric choice (1-4) or 'none' (case-insensitive, keyword-only);
    # sets CHOSEN_MODEL and CHOSEN_HF_REPO. Returns 1 for empty/invalid input
    # (caller must re-prompt — no silent default is assumed here).
    _norm=$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')
    case "$_norm" in
        1) CHOSEN_MODEL="nomic-embed-text-v1.5"; CHOSEN_HF_REPO="nomic-ai/nomic-embed-text-v1.5" ;;
        2) CHOSEN_MODEL="bge-small-en-v1.5";     CHOSEN_HF_REPO="BAAI/bge-small-en-v1.5" ;;
        3) CHOSEN_MODEL="bge-m3";                CHOSEN_HF_REPO="BAAI/bge-m3" ;;
        4) CHOSEN_MODEL="embeddinggemma-300m";   CHOSEN_HF_REPO="google/embeddinggemma-300m" ;;
        none) CHOSEN_MODEL="none";               CHOSEN_HF_REPO="" ;;
        *) return 1 ;;
    esac
    return 0
}

_show_menu() {
    echo ""
    echo "==> Select embedding model:"
    echo "    1) nomic-embed-text-v1.5   (nomic-ai/nomic-embed-text-v1.5)  [recommended]"
    echo "       dim=768, ~270MB, English-focused, balanced quality/speed, open"
    echo "    2) bge-small-en-v1.5       (BAAI/bge-small-en-v1.5)"
    echo "       dim=384, ~130MB, English-only, fastest/smallest, open"
    echo "    3) bge-m3                  (BAAI/bge-m3)"
    echo "       dim=1024, ~2.2GB, multilingual, highest quality/slower, open"
    echo "    4) embeddinggemma-300m     (google/embeddinggemma-300m)       [gated]"
    echo "       dim=768, ~300MB, multilingual, high quality, GATED (HF login)"
    echo "    none) keyword-only BM25 - no download, reduced semantic recall"
    echo ""
    if [ -n "$SAVED_MODEL" ]; then
        echo "    Current saved model: $SAVED_MODEL"
    fi
}

CHOSEN_MODEL=""
CHOSEN_HF_REPO=""

if [ -t 0 ]; then
    # Interactive stdin — loop until a valid choice. Empty/invalid input
    # re-prompts; there is no auto-accepted default. 'none' (keyword-only)
    # is always a valid answer and resolves the loop immediately.
    _show_menu
    while true; do
        printf "    Enter number (1-4), or 'none' for keyword-only (no embedding model): "
        if ! read -r _input; then
            echo "" >&2
            echo "ERROR: no input received (stdin closed); cannot proceed without a model choice." >&2
            exit 1
        fi
        if _resolve_model "$_input"; then
            break
        fi
        echo "    Invalid choice '$_input'. Please enter a number 1-4, or 'none'."
    done
else
    # Non-interactive stdin — honor a previously saved choice only. Never
    # silently substitute a specific model when unset/unrecognized: resolve
    # cleanly to keyword-only mode instead (informational, exit 0).
    case "$SAVED_MODEL" in
        nomic-embed-text-v1.5) CHOSEN_MODEL="nomic-embed-text-v1.5"; CHOSEN_HF_REPO="nomic-ai/nomic-embed-text-v1.5" ;;
        bge-small-en-v1.5)     CHOSEN_MODEL="bge-small-en-v1.5";     CHOSEN_HF_REPO="BAAI/bge-small-en-v1.5" ;;
        bge-m3)                CHOSEN_MODEL="bge-m3";                CHOSEN_HF_REPO="BAAI/bge-m3" ;;
        embeddinggemma-300m)   CHOSEN_MODEL="embeddinggemma-300m";   CHOSEN_HF_REPO="google/embeddinggemma-300m" ;;
        *)
            CHOSEN_MODEL="none"
            CHOSEN_HF_REPO=""
            ;;
    esac

    if [ "$CHOSEN_MODEL" = "none" ]; then
        echo "==> No embedding model configured — running in keyword-only mode (QUIPU_EMBEDDING_MODEL=none)."
        echo "    To use semantic search instead, set QUIPU_EMBEDDING_MODEL=<key> (e.g. nomic-embed-text-v1.5) before installing."
    else
        echo "Using model: $CHOSEN_MODEL"
    fi
fi

MODEL_DIR="$QUIPU_HOME/models/$CHOSEN_MODEL"

# Persist chosen model to config (idempotent, atomic)
if [ -f "$CONFIG_FILE" ]; then
    grep -v '^MODEL=' "$CONFIG_FILE" > "$CONFIG_FILE.tmp" || true
    printf 'MODEL=%s\n' "$CHOSEN_MODEL" >> "$CONFIG_FILE.tmp"
    mv "$CONFIG_FILE.tmp" "$CONFIG_FILE"
else
    mkdir -p "$(dirname "$CONFIG_FILE")"
    printf 'MODEL=%s\n' "$CHOSEN_MODEL" > "$CONFIG_FILE"
fi

# Test hook — used by tests/scripts/test_install_model_select.py to exercise
# the model-selection loop above without running the full install (venv/pip/
# network). Not used by real installs.
if [ "${QUIPU_TEST_MODEL_SELECT_ONLY:-0}" = "1" ]; then
    echo "CHOSEN_MODEL=$CHOSEN_MODEL"
    exit 0
fi

# Step 2 — repo root is parent of scripts/
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "==> Quipu install"
echo "    QUIPU_HOME : $QUIPU_HOME"
echo "    VENV       : $VENV"
echo "    REPO_ROOT  : $REPO_ROOT"

# Step 3 — create venv if absent
if [ ! -d "$VENV" ]; then
    echo "==> Creating venv at $VENV"
    python3 -m venv "$VENV"
else
    echo "==> Venv already present at $VENV"
fi

# Resolve python executable (POSIX primary; Windows/git-bash fallback)
if [ -f "$VENV/bin/python" ]; then
    PY="$VENV/bin/python"
elif [ -f "$VENV/Scripts/python.exe" ]; then
    PY="$VENV/Scripts/python.exe"
else
    echo "ERROR: cannot find python in venv $VENV" >&2
    exit 1
fi

# Step 4 — upgrade pip and install quipu (editable)
echo "==> Upgrading pip"
"$PY" -m pip install --upgrade pip

echo "==> Installing quipu (editable) from $REPO_ROOT"
# onnxruntime may fail on Python 3.14+; that's acceptable — CLI+store still work.
if ! "$PY" -m pip install -e "$REPO_ROOT"; then
    echo "WARNING: pip install returned non-zero (likely onnxruntime wheel unavailable)." >&2
    echo "WARNING: Installing non-onnxruntime deps (mcp, tokenizers, numpy) explicitly, then quipu with --no-deps (skips ALL declared deps including onnxruntime)." >&2
    # Install the runtime deps quipu needs for CLI/server to function (excludes onnxruntime).
    if ! "$PY" -m pip install mcp tokenizers numpy; then
        echo "WARNING: runtime deps install failed; CLI may be broken." >&2
    fi
    # Install quipu package itself without pulling in any declared deps (they were handled above).
    if ! "$PY" -m pip install -e "$REPO_ROOT" --no-deps; then
        echo "WARNING: quipu editable install failed even with --no-deps; continuing (CLI will be broken)." >&2
    fi
fi

# Step 5 — ensure huggingface_hub in venv (best-effort)
echo "==> Checking huggingface_hub"
if ! "$PY" -c 'import huggingface_hub' 2>/dev/null; then
    echo "==> Installing huggingface_hub into venv"
    "$PY" -m pip install huggingface_hub || echo "WARNING: huggingface_hub install failed; model fetch may not work." >&2
else
    echo "    huggingface_hub already present"
fi

# Step 6 — fetch chosen model if absent
if [ "$CHOSEN_MODEL" = "none" ]; then
    echo "==> Keyword-only mode (QUIPU_EMBEDDING_MODEL=none) — no model to download."
elif [ -f "$MODEL_DIR/model.onnx" ]; then
    echo "==> Model already present at $MODEL_DIR (skipping download)"
else
    echo "==> Fetching $CHOSEN_MODEL to $MODEL_DIR"
    mkdir -p "$MODEL_DIR"

    if "$PY" -c 'import huggingface_hub' 2>/dev/null; then
        if ! "$PY" -c "
from huggingface_hub import snapshot_download
snapshot_download('$CHOSEN_HF_REPO', local_dir='$MODEL_DIR', local_dir_use_symlinks=False, resume_download=True)
"; then
            echo "" >&2
            if [ "$CHOSEN_MODEL" = "embeddinggemma-300m" ]; then
                echo "WARNING: model download failed (likely gated — see below)." >&2
                echo "" >&2
                echo "  EmbeddingGemma-300m is a gated model and requires:" >&2
                echo "  1. Accept the license at https://huggingface.co/google/embeddinggemma-300m" >&2
                echo "  2. Run: hf auth login   (token from https://huggingface.co/settings/tokens)" >&2
                echo "  3. Re-run this installer." >&2
                echo "" >&2
                echo "  To fetch manually after login:" >&2
                echo "  hf download google/embeddinggemma-300m --local-dir \"$MODEL_DIR\"" >&2
                echo "" >&2
            else
                echo "WARNING: model download failed." >&2
                echo "  Check your internet connection and retry the installer." >&2
                echo "  To fetch manually: hf download $CHOSEN_HF_REPO --local-dir \"$MODEL_DIR\"" >&2
                echo "" >&2
            fi
        fi
    else
        echo "WARNING: huggingface_hub not installed; model download skipped." >&2
        echo "  Install: pip install huggingface_hub" >&2
        echo "  Then: hf download $CHOSEN_HF_REPO --local-dir \"$MODEL_DIR\"" >&2
    fi
fi

# Step 7 — validate
echo "==> Validating install"
"$PY" -m quipu --version || { echo "ERROR: quipu --version failed after install" >&2; exit 1; }

# Step 8 — echo .mcp.json hint
echo ""
echo "==> Install complete."
echo ""
echo "Add this to your .mcp.json to register Quipu with an MCP client:"
echo ""
cat <<SNIPPET
{
  "command": "$PY",
  "args": ["-m", "quipu", "serve"],
  "env": {
    "QUIPU_MODE": "project",
    "QUIPU_PROJECT_ROOT": "<your-project-root>",
    "QUIPU_EMBEDDING_MODEL": "$CHOSEN_MODEL"
  }
}
SNIPPET
echo ""
echo "For global mode, omit QUIPU_PROJECT_ROOT and set QUIPU_MODE=global."
echo "Run \`$PY -m quipu init\` inside your project to initialise the store."

# Step 9 — register native Claude Code hooks (opt-in, --with-hooks only)
if [ "$WITH_HOOKS" = "1" ]; then
    echo ""
    echo "==> Registering native Claude Code hooks (--with-hooks)"
    SETTINGS_JSON="$HOME/.claude/settings.json"
    # Pass REPO_ROOT via environment; Python uses os.environ to get it and
    # json.dump to encode it properly — handles spaces/metacharacters.
    QUIPU_HOOK_REPO_ROOT="$REPO_ROOT"
    export QUIPU_HOOK_REPO_ROOT

    "$PY" - "$SETTINGS_JSON" <<'PYEOF'
import json, sys, os, tempfile

settings_path = sys.argv[1]

# Build the command string from the env var so json.dump handles all escaping.
repo_root = os.environ.get("QUIPU_HOOK_REPO_ROOT", "")
capture_script = os.path.join(repo_root, "core", "hooks", "quipu-capture.sh")
# Use a forward-slash path for portability in the command string.
capture_script = capture_script.replace("\\", "/")
capture_cmd = "bash " + json.dumps(capture_script)

hook_entry = {"type": "command", "command": capture_cmd}
events = ["SessionStart", "PreCompact", "Stop", "UserPromptSubmit"]

# Load existing settings or start fresh
if os.path.exists(settings_path):
    try:
        with open(settings_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"quipu install: WARNING: {settings_path} is not valid JSON ({exc}).", file=sys.stderr)
        print(f"quipu install: Add the following snippet to {settings_path} manually:", file=sys.stderr)
        snippet = {"hooks": {e: [{"matcher": "", "hooks": [hook_entry]}] for e in events}}
        print(json.dumps(snippet, indent=2), file=sys.stderr)
        sys.exit(0)
else:
    data = {}
    os.makedirs(os.path.dirname(settings_path), exist_ok=True)

if not isinstance(data, dict):
    data = {}

hooks = data.setdefault("hooks", {})
if not isinstance(hooks, dict):
    hooks = {}
    data["hooks"] = hooks

for event in events:
    event_hooks = hooks.setdefault(event, [])
    if not isinstance(event_hooks, list):
        event_hooks = []
        hooks[event] = event_hooks
    # Find or create a catch-all matcher group.
    # We look for an existing group with matcher="" and append to its hooks array
    # if our command isn't already there. Otherwise we add a new group.
    found = False
    for group in event_hooks:
        if not isinstance(group, dict):
            continue
        if group.get("matcher", "") == "":
            inner = group.setdefault("hooks", [])
            if not isinstance(inner, list):
                inner = []
                group["hooks"] = inner
            # Idempotent: only add if not already present
            already = any(
                isinstance(h, dict) and h.get("command") == capture_cmd
                for h in inner
            )
            if not already:
                inner.append(hook_entry)
            found = True
            break
    if not found:
        event_hooks.append({"matcher": "", "hooks": [hook_entry]})

# Atomic write: write to temp file then os.replace onto settings.json.
settings_dir = os.path.dirname(settings_path)
tmp_fd, tmp_path = tempfile.mkstemp(dir=settings_dir, suffix=".tmp")
try:
    with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    os.replace(tmp_path, settings_path)
except Exception:
    try:
        os.unlink(tmp_path)
    except OSError:
        pass
    raise

print(f"quipu install: hooks registered in {settings_path}")
PYEOF

fi
