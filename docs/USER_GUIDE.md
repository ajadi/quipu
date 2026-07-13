# Quipu User Guide

## Installation

### From PyPI (recommended)

```bash
pip install quipu-mcp
pip install huggingface_hub    # required for automatic model download
```

No git clone, no install scripts needed.

### From a local checkout

```bash
cd path/to/quipu
pip install -e ".[test]"
```

### Initialise a project store

```bash
cd your-project
quipu init
```

This creates `.quipu/quipu.db` and `.quipu/config.json` in your project root,
and prints the exact `.mcp.json` snippet for your platform.

**Modes:**

| Mode | Command | DB location |
|------|---------|-------------|
| Project (default) | `quipu init` | `<cwd>/.quipu/quipu.db` |
| Global | `quipu init --mode global` | `~/.quipu/global.db` |
| Server (synced) | `quipu init --mode server` | `<cwd>/.quipu/quipu.db` + hub config |

---

## Embedding models

Quipu ships with 4 selectable ONNX embedding models. After you select a model,
it is downloaded **automatically on first use** (lazy download). No separate
download step needed.

### Available models

| Key | Size | HF repo | Gated |
|-----|------|---------|-------|
| `nomic-embed-text-v1.5` *(recommended)* | ~1 GB | `nomic-ai/nomic-embed-text-v1.5` | No |
| `bge-small-en-v1.5` | ~130 MB | `BAAI/bge-small-en-v1.5` | No |
| `bge-m3` | ~580 MB | `BAAI/bge-m3` | No |
| `embeddinggemma-300m` | ~1.2 GB | `google/embeddinggemma-300m` | **Yes** |

### Selecting a model

Set `QUIPU_EMBEDDING_MODEL` in your `.mcp.json`:

```json
{
  "mcpServers": {
    "quipu": {
      "command": "python",
      "args": ["-m", "quipu", "serve"],
      "env": {
        "QUIPU_MODE": "project",
        "QUIPU_PROJECT_ROOT": "/path/to/project",
        "QUIPU_EMBEDDING_MODEL": "bge-m3"
      }
    }
  }
}
```

The model downloads automatically the first time `quipu_write` or `quipu_search`
is called. Progress is printed to stderr.

### Gated models

`embeddinggemma-300m` requires:
1. Accept the license at https://huggingface.co/google/embeddinggemma-300m
2. `pip install huggingface_hub && huggingface-cli login`
3. Set `QUIPU_EMBEDDING_MODEL=embeddinggemma-300m`

### Model cache location

Models are stored at `~/.quipu/models/<model-key>/`. Override with
`QUIPU_MODEL_DIR=/custom/path`.

---

## Connecting to MCP clients

### Claude Code

Add to `.mcp.json` in your project root:

```json
{
  "mcpServers": {
    "quipu": {
      "command": "python",
      "args": ["-m", "quipu", "serve"],
      "env": {
        "QUIPU_MODE": "project",
        "QUIPU_PROJECT_ROOT": "."
      }
    }
  }
}
```

Run `quipu init` first — it prints the exact snippet with absolute paths.

### Cursor

Same `.mcp.json` format. Add it to your project or `~/.cursor/mcp.json` for
global availability.

### Global mode (shared across all projects)

```json
{
  "mcpServers": {
    "quipu": {
      "command": "python",
      "args": ["-m", "quipu", "serve"],
      "env": { "QUIPU_MODE": "global" }
    }
  }
}
```

### Any MCP client

Quipu is a standard MCP stdio server. Point any MCP-compatible client at
`python -m quipu serve` with the appropriate env vars.

### AI-guided setup

Tell your AI assistant: *"set up Quipu for me"* — it can run the installer
and wire up the `.mcp.json` entry.

---

## MCP tools reference

| Tool | Description |
|------|-------------|
| `quipu_write` | Store a memory atom. Returns detected conflicts (near-duplicates ≥0.92 cosine) for caller adjudication via `supersede` or `force`. |
| `quipu_search` | Multi-tier retrieval: exact match → cosine/sqlite-vec → FTS5 BM25 + reciprocal rank fusion. Supports `project_id`, `session_id`, `tags`, `kind`, `top_k`, `graph_expand`. |
| `quipu_get` | Fetch a single atom by ID. |
| `quipu_list` | List atoms for a project, newest first. |
| `quipu_invalidate` | Soft-delete an atom by ID. |
| `quipu_flush` | Optional Haiku enrichment (requires `ANTHROPIC_API_KEY`) + trigger push sync. |
| `quipu_stats` | Record counts, last-flush timestamp, sync status. |
| `quipu_push` | Push local changes to the sync hub (best-effort; offline-safe). |
| `quipu_pull` | Pull remote changes from the sync hub (best-effort; offline-safe). |
| `quipu_prime` | Session-start auto-recall — surfaces the most relevant prior memories. Call once at session start. Never raises; degrades gracefully. |
| `quipu_receipts` | Export hashed/redacted oplog audit for privacy-safe verification. |

---

## CLI commands

```
quipu --version
quipu init [--mode {project,global,server}]
quipu serve
quipu mirror --project-id <id> [--output-dir memory] [--db-path PATH]
quipu drain [--queue-path PATH] [--db-path PATH] [--project-id ID]
quipu backfill [--db-path PATH] [--project-id ID]
quipu receipts [--db-path PATH] [--project-id ID] [--limit N] [--format json|text]
quipu gc [--db-path PATH] [--project-id ID] [--apply] [--min-age-days 90] [--min-access-count 3]
```

| Command | Purpose |
|---------|---------|
| `init` | Create DB and config. Idempotent: re-running preserves `project_id`, refreshes metadata. Prints `.mcp.json` snippet. |
| `serve` | Start MCP server over stdio (register this in `.mcp.json`). |
| `mirror` | Render atoms to Markdown files under `memory/`. |
| `drain` | Drain capture queue into the store. |
| `backfill` | Re-emit pre-existing atoms into the oplog for sync (one-shot, idempotent). |
| `receipts` | Export privacy-safe hashed/redacted oplog audit. |
| `gc` | Preview stale, low-access atoms; add `--apply` to soft-invalidate them. |

---

## Environment variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `QUIPU_MODE` | `project` or `global` | `project` |
| `QUIPU_PROJECT_ROOT` | Project root for project-mode DB | `cwd` |
| `QUIPU_DB_PATH` | Explicit DB path (overrides mode routing) | — |
| `QUIPU_EMBEDDING_MODEL` | Active embedding model (`nomic-embed-text-v1.5`, `bge-small-en-v1.5`, `bge-m3`, `embeddinggemma-300m`); `none` disables embeddings | unset (`none`) |
| `QUIPU_MODEL_DIR` | Override ONNX model cache directory | `~/.quipu/models/<model>/` |
| `QUIPU_INVALIDATION_THRESHOLD` | Cosine threshold for conflict detection (0–1] | `0.92` |
| `ANTHROPIC_API_KEY` | Enables `quipu_flush` enrichment via Claude Haiku | — |
| `QUIPU_HUB_URL` | Sync hub URL (server mode) | — |
| `QUIPU_HUB_TOKEN` | Sync hub bearer token (server mode) | — |
| `QUIPU_MODEL_SHA256` | Optional integrity check — SHA-256 hex digest of model.onnx | — |

---

## Encrypted sync (optional)

Quipu supports zero-knowledge cross-machine sync via a self-hosted hub. The hub
stores only AES-256-GCM encrypted blobs — it never sees plaintext content or
project IDs (HMAC-blinded).

### Setup

```bash
# 1. Deploy the hub (Docker)
cd hub && docker compose up -d

# 2. Init a server-mode store
quipu init --mode server
# Prints client_id — save this.

# 3. Set env vars
export QUIPU_HUB_URL=http://your-hub:8000
export QUIPU_HUB_TOKEN=your-secure-token

# 4. Push existing memories to the hub
quipu backfill
quipu push
```

See `hub/deploy/RUNBOOK.md` for detailed deployment instructions.

---

## Quickstart (60 seconds from zero)

```bash
# 1. Install
pip install quipu-mcp
pip install huggingface_hub

# 2. Init your project
cd my-project
quipu init

# 3. Copy the printed .mcp.json snippet into your .mcp.json
# 4. Restart your MCP client
# 5. Tell your AI: "write this to Quipu: the deploy script is at scripts/deploy.sh"
```

After you select an embedding model, the first `quipu_write` or `quipu_search`
will automatically download it (~1 GB for the recommended
`nomic-embed-text-v1.5`). This is a one-time download.

---

## Troubleshooting

### "No module named 'huggingface_hub'"
```bash
pip install huggingface_hub
```

### Model download fails
Check your internet connection. For gated models (`embeddinggemma-300m`), ensure
you've run `huggingface-cli login`. Set `QUIPU_EMBEDDING_MODEL=nomic-embed-text-v1.5`
for an ungated model.

### "No module named 'onnxruntime'"
Windows ARM / Python 3.14+ may not have a prebuilt onnxruntime wheel.
Install from source or use Python 3.10–3.13.

### Tests fail on Windows
Shell-based capture tests are skipped on Windows. This is expected — they
require bash and Unix paths. The core storage, retrieval, write, and MCP
tests all pass on Windows.

### "Permission denied" on model download
Model cache directory is `~/.quipu/models/`. Ensure your user has write
permissions to `~/.quipu/`.
