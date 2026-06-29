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
    # $1 = numeric choice (1-5); sets CHOSEN_MODEL and CHOSEN_HF_REPO
    case "$1" in
        1) CHOSEN_MODEL="nomic-embed-v2";        CHOSEN_HF_REPO="nomic-ai/nomic-embed-v2" ;;
        2) CHOSEN_MODEL="nomic-embed-text-v1.5"; CHOSEN_HF_REPO="nomic-ai/nomic-embed-text-v1.5" ;;
        3) CHOSEN_MODEL="bge-small-en-v1.5";     CHOSEN_HF_REPO="BAAI/bge-small-en-v1.5" ;;
        4) CHOSEN_MODEL="bge-m3";                CHOSEN_HF_REPO="BAAI/bge-m3" ;;
        5) CHOSEN_MODEL="embeddinggemma-300m";   CHOSEN_HF_REPO="google/embeddinggemma-300m" ;;
        *) return 1 ;;
    esac
    return 0
}

_show_menu() {
    echo ""
    echo "==> Select embedding model:"
    echo "    1) nomic-embed-v2          (nomic-ai/nomic-embed-v2)         [recommended]"
    echo "    2) nomic-embed-text-v1.5   (nomic-ai/nomic-embed-text-v1.5)"
    echo "    3) bge-small-en-v1.5       (BAAI/bge-small-en-v1.5)"
    echo "    4) bge-m3                  (BAAI/bge-m3)"
    echo "    5) embeddinggemma-300m     (google/embeddinggemma-300m)       [gated]"
    echo ""
    if [ -n "$SAVED_MODEL" ]; then
        echo "    Current saved model: $SAVED_MODEL"
    fi
}

# Determine default number from saved model
_default_num="1"
if [ -n "$SAVED_MODEL" ]; then
    case "$SAVED_MODEL" in
        nomic-embed-v2)        _default_num="1" ;;
        nomic-embed-text-v1.5) _default_num="2" ;;
        bge-small-en-v1.5)     _default_num="3" ;;
        bge-m3)                _default_num="4" ;;
        embeddinggemma-300m)   _default_num="5" ;;
        *)                     _default_num="1" ;;
    esac
fi

CHOSEN_MODEL=""
CHOSEN_HF_REPO=""

if [ -t 0 ]; then
    # Interactive stdin — show menu
    _show_menu
    printf "    Enter number [default: %s]: " "$_default_num"
    read -r _input || true
    if [ -z "$_input" ]; then
        _choice="$_default_num"
    else
        _choice="$_input"
    fi

    if ! _resolve_model "$_choice"; then
        # Out of range — re-prompt once
        echo "    Invalid choice '$_choice'. Please enter a number 1-5."
        _show_menu
        printf "    Enter number [default: %s]: " "$_default_num"
        read -r _input2 || true
        if [ -z "$_input2" ]; then
            _choice2="$_default_num"
        else
            _choice2="$_input2"
        fi
        if ! _resolve_model "$_choice2"; then
            # Fall back to default
            _resolve_model "$_default_num"
        fi
    fi
else
    # Non-interactive stdin — use saved or default
    if [ -n "$SAVED_MODEL" ]; then
        case "$SAVED_MODEL" in
            nomic-embed-v2)        CHOSEN_MODEL="nomic-embed-v2";        CHOSEN_HF_REPO="nomic-ai/nomic-embed-v2" ;;
            nomic-embed-text-v1.5) CHOSEN_MODEL="nomic-embed-text-v1.5"; CHOSEN_HF_REPO="nomic-ai/nomic-embed-text-v1.5" ;;
            bge-small-en-v1.5)     CHOSEN_MODEL="bge-small-en-v1.5";     CHOSEN_HF_REPO="BAAI/bge-small-en-v1.5" ;;
            bge-m3)                CHOSEN_MODEL="bge-m3";                CHOSEN_HF_REPO="BAAI/bge-m3" ;;
            embeddinggemma-300m)   CHOSEN_MODEL="embeddinggemma-300m";   CHOSEN_HF_REPO="google/embeddinggemma-300m" ;;
            *)
                CHOSEN_MODEL="nomic-embed-v2"
                CHOSEN_HF_REPO="nomic-ai/nomic-embed-v2"
                ;;
        esac
    else
        CHOSEN_MODEL="nomic-embed-v2"
        CHOSEN_HF_REPO="nomic-ai/nomic-embed-v2"
    fi
    echo "Using model: $CHOSEN_MODEL"
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
if [ -f "$MODEL_DIR/model.onnx" ]; then
    echo "==> Model already present at $MODEL_DIR (skipping download)"
else
    echo "==> Fetching $CHOSEN_MODEL to $MODEL_DIR"
    mkdir -p "$MODEL_DIR"
    # Prefer venv huggingface-cli; fall back to PATH
    HF_CLI=""
    if [ -f "$VENV/bin/huggingface-cli" ]; then
        HF_CLI="$VENV/bin/huggingface-cli"
    elif [ -f "$VENV/Scripts/huggingface-cli.exe" ]; then
        HF_CLI="$VENV/Scripts/huggingface-cli.exe"
    elif command -v huggingface-cli >/dev/null 2>&1; then
        HF_CLI="huggingface-cli"
    fi

    if [ -n "$HF_CLI" ]; then
        if ! "$HF_CLI" download "$CHOSEN_HF_REPO" --local-dir "$MODEL_DIR"; then
            echo "" >&2
            if [ "$CHOSEN_MODEL" = "embeddinggemma-300m" ]; then
                echo "WARNING: model download failed (likely gated — see below)." >&2
                echo "" >&2
                echo "  EmbeddingGemma-300m is a gated model and requires:" >&2
                echo "  1. Accept the license at https://huggingface.co/google/embeddinggemma-300m" >&2
                echo "  2. Run: huggingface-cli login   (token from https://huggingface.co/settings/tokens)" >&2
                echo "  3. Re-run this installer." >&2
                echo "" >&2
                echo "  To fetch manually after login:" >&2
                echo "  huggingface-cli download google/embeddinggemma-300m --local-dir \"$MODEL_DIR\"" >&2
                echo "" >&2
            else
                echo "WARNING: model download failed." >&2
                echo "  Check your internet connection and retry the installer." >&2
                echo "  To fetch manually: huggingface-cli download $CHOSEN_HF_REPO --local-dir \"$MODEL_DIR\"" >&2
                echo "" >&2
            fi
        fi
    else
        echo "WARNING: huggingface-cli not found; skipping model download." >&2
        echo "  Run manually: pip install huggingface_hub" >&2
        echo "  Then: huggingface-cli download $CHOSEN_HF_REPO --local-dir \"$MODEL_DIR\"" >&2
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
