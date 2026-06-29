# Quipu Hub

Zero-knowledge relay service for Quipu oplog sync.  The hub stores opaque
ciphertext blobs partitioned by a blinded project ID.  It never holds a
decryption key, never sees plaintext, and never stores a real project identity.

---

## Quickstart — Container (Recommended / Default)

### Prerequisites

- Docker >= 20 (or Podman >= 4 with `podman-compose`)
- A domain name with DNS pointing at this host (for TLS)

### 1. Write hub.env

```sh
# Generate a bearer token
HUB_TOKEN=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
echo "HUB_TOKENS=${HUB_TOKEN}" > hub/hub.env
echo "Token: ${HUB_TOKEN}"   # share this with the Quipu client config
```

`hub.env` lives in `hub/` alongside `docker-compose.yml`.  It is gitignored.

### 2. Start the service

```sh
docker compose -f hub/docker-compose.yml up -d
```

### 3. Smoke test

```sh
curl http://localhost:8000/health
# {"status":"ok"}
```

---

## Alternative: Bare-metal / venv

Use this path if you cannot run Docker.

```sh
python -m venv hub/.venv
source hub/.venv/bin/activate          # Windows: hub\.venv\Scripts\activate
pip install -r hub/requirements.txt
export HUB_TOKENS="<your-token>"
uvicorn hub.main:app --workers 1
```

For production bare-metal use the systemd unit in
`hub/deploy/quipu-hub.service.example` and terminate TLS at a reverse proxy
(see TLS section below).

---

## TLS

The hub binds an unprivileged port (default 8000) and **delegates TLS to a
reverse proxy**.  This is the recommended and default posture.

**Reverse proxy (default):**

- Caddy (simplest): see commented block in `docker-compose.yml`.
- nginx + certbot: obtain a cert with `certbot --nginx -d hub.example.com`;
  configure `proxy_pass http://localhost:8000`.
- HTTP → HTTPS redirect is the proxy's responsibility.

**Direct TLS (advanced, no proxy):**

Pass cert paths to uvicorn:

```sh
uvicorn hub.main:app --workers 1 \
  --ssl-certfile /etc/letsencrypt/live/hub.example.com/fullchain.pem \
  --ssl-keyfile  /etc/letsencrypt/live/hub.example.com/privkey.pem
```

Or set `HUB_TLS_CERT` / `HUB_TLS_KEY` env vars and pass them in your launch
command.  The hub itself never reads these vars — they are documentation hooks
for your launch wrapper.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `HUB_TOKENS` | (required) | Comma-separated bearer tokens issued to clients. Hashed at startup; raw values never stored. |
| `HUB_DB_PATH` | `hub/hub.db` | Path to the SQLite database file. |
| `HUB_AUDIT_PATH` | `hub/audit.log` | Path to the append-only audit log. |
| `HUB_RATE_LIMIT` | `1000` | Max requests per token+project per rate window. |
| `HUB_RATE_WINDOW` | `3600` | Rate-limit window in seconds. |
| `HUB_MAX_BODY_BYTES` | `10485760` | Max POST body size in bytes (413 if exceeded). |
| `HUB_MAX_ENTRIES` | `1000` | Max entries per POST batch (413 if exceeded). |
| `HUB_TLS_CERT` | (unset) | Path to TLS certificate (informational; pass to uvicorn CLI). |
| `HUB_TLS_KEY` | (unset) | Path to TLS private key (informational; pass to uvicorn CLI). |

---

## Security Model

The hub is zero-knowledge: it stores opaque AES-256-GCM ciphertext produced
by the Quipu client and returns it verbatim.  The hub holds no decryption key,
no passphrase, and no real project identity — only a blinded pseudonym (a
64-hex HMAC-SHA256 derived by the client).  Bearer token auth is required on
every request; tokens are SHA-256 hashed at startup and compared in constant
time.  The hub never runs as root and never needs root at runtime.  Claude
never deploys or configures this service for you — all deployment steps are
performed manually by a human operator using the runbook in
`hub/deploy/RUNBOOK.md`.
