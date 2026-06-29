#!/usr/bin/env sh
# quipu-capture.sh — durable spool producer for Quipu pipeline events.
#
# Appends ONE JSON line to ${CLAUDE_PROJECT_DIR:-.}/.quipu/capture-queue.jsonl.
# The drain (spool → quipu_write) is owned by the mcp/modes track / E9.
# This hook is producer-only. See OQ-007a in tasks/TASK-007.md.
#
# INVOCATION PATTERNS (5 capture points):
#   1. pre-compact (automatic):
#      Called from core/hooks/pre-compact.sh with QUIPU_CAPTURE_SOURCE=pre_compact
#      and the snapshot file piped to stdin.
#
#   2. PM Step 13 close (explicit, PM calls this):
#      QUIPU_CAPTURE_SOURCE=pm_close \
#        QUIPU_TASK_ID=TASK-XXX \
#        bash "$CLAUDE_PROJECT_DIR/core/hooks/quipu-capture.sh" < summary_file
#
#   3. Reality-checker output (explicit):
#      QUIPU_CAPTURE_SOURCE=reality_check \
#        QUIPU_TASK_ID=TASK-XXX \
#        bash "$CLAUDE_PROJECT_DIR/core/hooks/quipu-capture.sh" < findings_file
#
#   4. OQ resolution (explicit):
#      QUIPU_CAPTURE_SOURCE=oq_resolution \
#        QUIPU_TASK_ID=TASK-XXX \
#        bash "$CLAUDE_PROJECT_DIR/core/hooks/quipu-capture.sh" < answer_file
#
#   5. Retrospective (explicit):
#      QUIPU_CAPTURE_SOURCE=retro \
#        QUIPU_TASK_ID=TASK-XXX \
#        bash "$CLAUDE_PROJECT_DIR/core/hooks/quipu-capture.sh" < retro_file
#
# SETTINGS REGISTRATION SNIPPET (paste into real .claude/settings.json during E9):
#   {
#     "hooks": {
#       "PreCompact": [
#         { "matcher": "", "hooks": [
#           { "type": "command",
#             "command": "bash \"$CLAUDE_PROJECT_DIR\"/core/hooks/pre-compact.sh" }
#         ]}
#       ]
#     }
#   }
#
# NOTE OQ-007a: drain (spool → quipu_write) is owned by the mcp/modes track / E9.
# This hook writes durably; zero data loss before drain is wired.
set +e

# Guard: QUIPU_PROJECT_ID must be set and non-empty.
if [ -z "${QUIPU_PROJECT_ID:-}" ]; then
    exit 0
fi

# Read stdin once (tolerate empty — direct invocations may pipe nothing).
# Guard with timeout if available so the hook never hangs when no stdin is attached.
if command -v timeout >/dev/null 2>&1; then
    INPUT=$(timeout 1 cat 2>/dev/null || true)
else
    INPUT=$(cat 2>/dev/null || true)
fi

# --- Derive fields ---

# QUIPU_CAPTURE_NO_JQ=1 forces the sed fallback path (used in tests / environments
# where jq is present but should be bypassed).
_HAS_JQ=0
if [ "${QUIPU_CAPTURE_NO_JQ:-0}" != "1" ] && command -v jq >/dev/null 2>&1; then
    _HAS_JQ=1
fi

# Derive SOURCE: use QUIPU_CAPTURE_SOURCE if set (all 5 Forge callers set it),
# else parse .hook_event_name from stdin JSON and map to a canonical source name.
if [ -n "${QUIPU_CAPTURE_SOURCE:-}" ]; then
    SOURCE="$QUIPU_CAPTURE_SOURCE"
else
    # Try to extract hook_event_name from stdin JSON.
    _HOOK_EVENT=""
    if [ "$_HAS_JQ" = "1" ]; then
        _HOOK_EVENT=$(printf '%s' "$INPUT" | jq -r '.hook_event_name // empty' 2>/dev/null)
    else
        _HOOK_EVENT=$(printf '%s' "$INPUT" | grep -oE '"hook_event_name"[[:space:]]*:[[:space:]]*"[^"]*"' \
            | head -1 | sed 's/.*:[[:space:]]*"//;s/"$//')
    fi
    case "$_HOOK_EVENT" in
        SessionStart)    SOURCE="session_start" ;;
        PreCompact)      SOURCE="pre_compact" ;;
        Stop)            SOURCE="stop" ;;
        UserPromptSubmit) SOURCE="user_prompt" ;;
        *)               SOURCE="unknown" ;;
    esac
fi

# Agent: from JSON .agent_type/.agent_name if present, else "unknown".
if [ "$_HAS_JQ" = "1" ]; then
    AGENT=$(printf '%s' "$INPUT" | jq -r '.agent_name // .agent_type // "unknown"' 2>/dev/null)
    [ -z "$AGENT" ] && AGENT="unknown"
else
    AGENT=$(printf '%s' "$INPUT" | grep -oE '"agent_(name|type)"[[:space:]]*:[[:space:]]*"[^"]*"' \
        | head -1 | sed 's/.*:[[:space:]]*"//;s/"$//')
    [ -z "$AGENT" ] && AGENT="unknown"
fi

# Task ID: from env QUIPU_TASK_ID first, then parse TASK-XXX token from stdin, else JSON null.
if [ -n "${QUIPU_TASK_ID:-}" ]; then
    TASK_ID="$QUIPU_TASK_ID"
else
    TASK_ID=$(printf '%s' "$INPUT" | grep -oE 'TASK-[0-9]+' | head -1)
fi

# Session ID: from env QUIPU_SESSION_ID first, else parse .session_id from stdin JSON, else empty.
if [ -n "${QUIPU_SESSION_ID:-}" ]; then
    SESSION_ID="$QUIPU_SESSION_ID"
else
    if [ "$_HAS_JQ" = "1" ]; then
        SESSION_ID=$(printf '%s' "$INPUT" | jq -r '.session_id // empty' 2>/dev/null)
    else
        SESSION_ID=$(printf '%s' "$INPUT" | grep -oE '"session_id"[[:space:]]*:[[:space:]]*"[^"]*"' \
            | head -1 | sed 's/.*:[[:space:]]*"//;s/"$//')
    fi
fi

# Timestamp (ISO-8601 UTC).
TS=$(date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || date +%Y-%m-%dT%H:%M:%SZ)

PROJECT_ID="$QUIPU_PROJECT_ID"

# --- Build content ---
# If stdin looks like Forge JSON (starts with '{'), try to extract a text field.
# Recognizable text fields: summary, content, text, output, message, description.
# Otherwise use raw stdin. Flatten newlines → space, strip CR.
CONTENT=""
if [ "$_HAS_JQ" = "1" ]; then
    PARSED=$(printf '%s' "$INPUT" | jq -r \
        '.summary // .content // .text // .output // .message // .description // .prompt // empty' \
        2>/dev/null)
    if [ -n "$PARSED" ]; then
        CONTENT="$PARSED"
    else
        CONTENT="$INPUT"
    fi
else
    # No jq: if input starts with '{' try grep for common text fields.
    FIRST_CHAR=$(printf '%s' "$INPUT" | head -c 1)
    if [ "$FIRST_CHAR" = "{" ]; then
        PARSED=$(printf '%s' "$INPUT" | grep -oE '"(summary|content|text|output|message|description|prompt)"[[:space:]]*:[[:space:]]*"([^"\\]|\\.)*"' \
            | head -1 | sed 's/^"[^"]*"[[:space:]]*:[[:space:]]*"//;s/"$//')
        if [ -n "$PARSED" ]; then
            CONTENT="$PARSED"
        else
            CONTENT="$INPUT"
        fi
    else
        CONTENT="$INPUT"
    fi
fi

# Flatten: strip CR, convert LF → space.
CONTENT=$(printf '%s' "$CONTENT" | tr -d '\r' | tr '\n' ' ')

# Length cap: truncate to 100 000 chars so a pathological stdin cannot bloat
# the queue beyond the drain's 1 MB per-line guard.
# ${#CONTENT} is POSIX sh. cut -c avoids subshell expansion limits on some
# shells, so we use it for the truncation itself.
_CONTENT_MAX=100000
if [ "${#CONTENT}" -gt "$_CONTENT_MAX" ]; then
    CONTENT=$(printf '%s' "$CONTENT" | cut -c1-${_CONTENT_MAX})
fi

# --- Build JSON line ---

SPOOL_DIR="${CLAUDE_PROJECT_DIR:-.}/.quipu"
SPOOL_FILE="${SPOOL_DIR}/capture-queue.jsonl"

# mkdir -p the spool dir.
if ! mkdir -p "$SPOOL_DIR" 2>/dev/null; then
    printf 'quipu-capture: warning: cannot create spool dir %s — capture skipped\n' "$SPOOL_DIR" >&2
    exit 0
fi

if [ "$_HAS_JQ" = "1" ]; then
    # Safe escaping via jq.
    if [ -n "$TASK_ID" ] && [ -n "$SESSION_ID" ]; then
        LINE=$(jq -cn \
            --arg source     "$SOURCE" \
            --arg agent      "$AGENT" \
            --arg task_id    "$TASK_ID" \
            --arg proj       "$PROJECT_ID" \
            --arg ts         "$TS" \
            --arg content    "$CONTENT" \
            --arg session_id "$SESSION_ID" \
            '{v:1,source:$source,agent:$agent,task_id:$task_id,session_id:$session_id,project_id:$proj,ts:$ts,content:$content,metadata:{source:$source,agent:$agent,task_id:$task_id,captured_by:"quipu-capture.sh"}}' \
            2>/dev/null)
    elif [ -n "$TASK_ID" ]; then
        LINE=$(jq -cn \
            --arg source  "$SOURCE" \
            --arg agent   "$AGENT" \
            --arg task_id "$TASK_ID" \
            --arg proj    "$PROJECT_ID" \
            --arg ts      "$TS" \
            --arg content "$CONTENT" \
            '{v:1,source:$source,agent:$agent,task_id:$task_id,session_id:null,project_id:$proj,ts:$ts,content:$content,metadata:{source:$source,agent:$agent,task_id:$task_id,captured_by:"quipu-capture.sh"}}' \
            2>/dev/null)
    elif [ -n "$SESSION_ID" ]; then
        LINE=$(jq -cn \
            --arg source     "$SOURCE" \
            --arg agent      "$AGENT" \
            --arg proj       "$PROJECT_ID" \
            --arg ts         "$TS" \
            --arg content    "$CONTENT" \
            --arg session_id "$SESSION_ID" \
            '{v:1,source:$source,agent:$agent,task_id:null,session_id:$session_id,project_id:$proj,ts:$ts,content:$content,metadata:{source:$source,agent:$agent,task_id:null,captured_by:"quipu-capture.sh"}}' \
            2>/dev/null)
    else
        LINE=$(jq -cn \
            --arg source  "$SOURCE" \
            --arg agent   "$AGENT" \
            --arg proj    "$PROJECT_ID" \
            --arg ts      "$TS" \
            --arg content "$CONTENT" \
            '{v:1,source:$source,agent:$agent,task_id:null,session_id:null,project_id:$proj,ts:$ts,content:$content,metadata:{source:$source,agent:$agent,task_id:null,captured_by:"quipu-capture.sh"}}' \
            2>/dev/null)
    fi
else
    # jq absent: sed-based escaper for string fields.
    # Order: strip control bytes → escape backslash → escape quote → escape TAB.
    # tr ranges: 0x00-0x08, 0x0B, 0x0C, 0x0E-0x1F (CR=0x0D stripped by tr -d '\r' above;
    # LF=0x0A flattened to space above; TAB=0x09 escaped below).
    _TAB=$(printf '\t')
    _esc() {
        printf '%s' "$1" | tr -d '\000-\010\013\014\016-\037' | sed "s/\\\\/\\\\\\\\/g; s/\"/\\\\\"/g; s/${_TAB}/\\\\t/g"
    }
    E_SOURCE=$(_esc "$SOURCE")
    E_AGENT=$(_esc "$AGENT")
    E_PROJ=$(_esc "$PROJECT_ID")
    E_TS=$(_esc "$TS")
    E_CONTENT=$(_esc "$CONTENT")

    if [ -n "$TASK_ID" ]; then
        E_TASK=$(_esc "$TASK_ID")
        TASK_JSON="\"${E_TASK}\""
        META_TASK="\"task_id\":\"${E_TASK}\""
    else
        TASK_JSON="null"
        META_TASK="\"task_id\":null"
    fi

    if [ -n "$SESSION_ID" ]; then
        E_SESSION=$(_esc "$SESSION_ID")
        SESSION_JSON="\"${E_SESSION}\""
    else
        SESSION_JSON="null"
    fi

    LINE="{\"v\":1,\"source\":\"${E_SOURCE}\",\"agent\":\"${E_AGENT}\",\"task_id\":${TASK_JSON},\"session_id\":${SESSION_JSON},\"project_id\":\"${E_PROJ}\",\"ts\":\"${E_TS}\",\"content\":\"${E_CONTENT}\",\"metadata\":{\"source\":\"${E_SOURCE}\",\"agent\":\"${E_AGENT}\",${META_TASK},\"captured_by\":\"quipu-capture.sh\"}}"
fi

if [ -z "$LINE" ]; then
    printf 'quipu-capture: warning: failed to build JSON line — capture skipped\n' >&2
    exit 0
fi

# Append to spool (single atomic printf >> ; cheap, well under 2s).
if ! printf '%s\n' "$LINE" >> "$SPOOL_FILE" 2>/dev/null; then
    printf 'quipu-capture: warning: cannot write to spool %s — capture skipped\n' "$SPOOL_FILE" >&2
    exit 0
fi

exit 0
