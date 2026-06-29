# Quipu Hub — Operator Runbook

Manual, copy-paste runbook.  Assumes: shell access to the target host, no
root, no Claude on the server, no pre-existing credentials.

---

## Path A — Container (Recommended / Default)

### Step 1: Generate a bearer token

```sh
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Save the output.  You will give this value to your Quipu client.

### Step 2: Write hub.env

```sh
cat > hub/hub.env <<'EOF'
HUB_TOKENS=<paste-token-here>
# Optional overrides:
# HUB_DB_PATH=/data/hub.db
# HUB_AUDIT_PATH=/audit/audit.log
# HUB_RATE_LIMIT=1000
# HUB_RATE_WINDOW=3600
EOF
```

`hub.env` must not be committed to git (it is gitignored by default).

### Step 3: Build and start the container

```sh
docker compose -f hub/docker-compose.yml up -d --build
```

### Step 4: Configure TLS reverse proxy

Choose one:

**Caddy (automatic Let's Encrypt):**

```sh
cat > hub/Caddyfile <<'EOF'
hub.example.com {
    reverse_proxy localhost:8000
}
EOF
# Add a caddy service to docker-compose.yml (see commented block) then:
docker compose -f hub/docker-compose.yml up -d
```

**nginx + certbot (manual cert):**

```sh
certbot --nginx -d hub.example.com
# nginx config: proxy_pass http://127.0.0.1:8000;
```

HTTP to HTTPS redirect is managed by the proxy.

### Step 5: Verify health

```sh
curl https://hub.example.com/health
# {"status":"ok"}
```

### Step 6: Full authed round-trip

```sh
TOKEN="<your-token>"
BPID="$(python -c "import secrets; print(secrets.token_hex(32))")"

# Push an entry
PAYLOAD_B64="$(python -c "import base64; print(base64.b64encode(b'opaque-ciphertext').decode())")"
curl -s -X POST "https://hub.example.com/oplog/${BPID}" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d "{\"entries\":[{
    \"entry_id\": \"$(python -c "import secrets; print(secrets.token_hex(32))")\",
    \"client_id\": \"client-1\",
    \"sequence_no\": 1,
    \"op\": \"upsert\",
    \"record_id\": \"rec-1\",
    \"blinded_project_id\": \"${BPID}\",
    \"ts\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",
    \"payload\": \"${PAYLOAD_B64}\"
  }]}"
# {"cursor":"1"}

# Pull entries since cursor 0
curl -s "https://hub.example.com/oplog/${BPID}?since=0" \
  -H "Authorization: Bearer ${TOKEN}"
# {"entries":[{...,"payload":"<same-base64>"}],"cursor":"1"}
```

Confirm the `payload` field in the pull response matches `${PAYLOAD_B64}`.

---

## Path B — Bare-metal / venv + systemd (Alternative)

### Step 1: Generate token (same as above)

### Step 2: Create venv and install deps

```sh
python -m venv hub/.venv
source hub/.venv/bin/activate
pip install -r hub/requirements.txt
```

### Step 3: Write /etc/quipu-hub.env

```sh
sudo tee /etc/quipu-hub.env > /dev/null <<'EOF'
HUB_TOKENS=<paste-token-here>
HUB_DB_PATH=/var/lib/quipu-hub/hub.db
HUB_AUDIT_PATH=/var/log/quipu-hub/audit.log
EOF
sudo chmod 600 /etc/quipu-hub.env
```

### Step 4: Create the system user and directories

```sh
sudo useradd --system --no-create-home --shell /usr/sbin/nologin quipu-hub
sudo mkdir -p /var/lib/quipu-hub /var/log/quipu-hub
sudo chown quipu-hub: /var/lib/quipu-hub /var/log/quipu-hub
```

### Step 5: Install the systemd unit

```sh
sudo cp hub/deploy/quipu-hub.service.example /etc/systemd/system/quipu-hub.service
# Edit WorkingDirectory and ExecStart paths if needed
sudo systemctl daemon-reload
sudo systemctl enable --now quipu-hub
```

### Step 6: Configure TLS proxy (same as Path A Step 4)

### Step 7: Verify health and round-trip (same as Path A Steps 5-6)
