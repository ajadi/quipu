#!/usr/bin/env sh
# hub/deploy/setup.sh — automated Quipu Hub bootstrap.
# POSIX sh, set -eu.  Never requires root.  Never bakes credentials.  Idempotent.
#
# Path A (default): Docker or Podman detected -> build image + compose up -d
# Path B (fallback): venv + pip install, then prints exact launch command.
#
# Usage:
#   export HUB_TOKENS="<token>"        # OR provide --env-file hub/hub.env
#   sh hub/deploy/setup.sh [--env-file <path>]

set -eu

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
ENV_FILE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --env-file)
            ENV_FILE="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Load env-file if provided
# ---------------------------------------------------------------------------
if [ -n "$ENV_FILE" ]; then
    if [ ! -f "$ENV_FILE" ]; then
        echo "ERROR: env-file not found: $ENV_FILE" >&2
        exit 1
    fi
    # shellcheck disable=SC1090
    . "$ENV_FILE"
fi

# ---------------------------------------------------------------------------
# Guard: HUB_TOKENS must be set
# ---------------------------------------------------------------------------
if [ -z "${HUB_TOKENS:-}" ]; then
    echo "" >&2
    echo "ERROR: HUB_TOKENS is not set." >&2
    echo "" >&2
    echo "Generate a token:" >&2
    echo "  python -c \"import secrets; print(secrets.token_urlsafe(32))\"" >&2
    echo "" >&2
    echo "Then either:" >&2
    echo "  export HUB_TOKENS=<token> && sh hub/deploy/setup.sh" >&2
    echo "  sh hub/deploy/setup.sh --env-file hub/hub.env" >&2
    echo "" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Locate project root (parent of the hub/ directory)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
HUB_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${HUB_DIR}/.." && pwd)"

# ---------------------------------------------------------------------------
# Path A: container runtime detected
# ---------------------------------------------------------------------------
detect_compose() {
    if command -v docker >/dev/null 2>&1; then
        if docker compose version >/dev/null 2>&1; then
            echo "docker compose"
            return
        fi
    fi
    if command -v podman-compose >/dev/null 2>&1; then
        echo "podman-compose"
        return
    fi
    echo ""
}

COMPOSE_CMD="$(detect_compose)"

if [ -n "${COMPOSE_CMD}" ]; then
    echo "==> Container runtime detected: ${COMPOSE_CMD}"
    echo "==> Building and starting quipu-hub..."

    # Write hub.env if it does not already exist
    HUB_ENV_FILE="${HUB_DIR}/hub.env"
    if [ ! -f "${HUB_ENV_FILE}" ]; then
        printf 'HUB_TOKENS=%s\n' "${HUB_TOKENS}" > "${HUB_ENV_FILE}"
        echo "    Written: ${HUB_ENV_FILE}"
    else
        echo "    Skipping hub.env (already exists): ${HUB_ENV_FILE}"
    fi

    COMPOSE_FILE="${HUB_DIR}/docker-compose.yml"
    cd "${PROJECT_ROOT}"
    ${COMPOSE_CMD} -f "${COMPOSE_FILE}" up -d --build

    echo ""
    echo "==> quipu-hub is running."
    echo "    Smoke test: curl http://localhost:8000/health"
    echo "    See hub/deploy/RUNBOOK.md for TLS configuration."
    exit 0
fi

# ---------------------------------------------------------------------------
# Path B: no container runtime — venv fallback
# ---------------------------------------------------------------------------
echo "==> No container runtime found. Using venv fallback."

VENV_DIR="${HUB_DIR}/.venv"

if [ -d "${VENV_DIR}" ]; then
    echo "    venv already exists, skipping creation: ${VENV_DIR}"
else
    echo "    Creating venv: ${VENV_DIR}"
    python -m venv "${VENV_DIR}"
fi

PIP="${VENV_DIR}/bin/pip"
if [ ! -x "${PIP}" ]; then
    # Windows path
    PIP="${VENV_DIR}/Scripts/pip"
fi

echo "    Installing dependencies..."
"${PIP}" install --quiet -r "${HUB_DIR}/requirements.txt"

echo ""
echo "==> Setup complete (venv path)."
echo ""
echo "    To start the hub, run:"
echo ""
echo "      export HUB_TOKENS='${HUB_TOKENS}'"
echo "      export HUB_DB_PATH='${HUB_DIR}/hub.db'"
echo "      export HUB_AUDIT_PATH='${HUB_DIR}/audit.log'"
echo "      ${VENV_DIR}/bin/uvicorn hub.main:app --workers 1"
echo ""
echo "    For production: use the systemd unit in hub/deploy/quipu-hub.service.example"
echo "    and terminate TLS at a reverse proxy. See hub/deploy/RUNBOOK.md."
