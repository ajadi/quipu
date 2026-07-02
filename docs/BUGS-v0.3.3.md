# v0.3.3 QA Report — Bugs Found

## CRITICAL

### BUG-1: `_MODELS_DIRECT` typo in loader.py
**File:** `quipu/models/loader.py:35`
`MODELS_DIRECT` referenced without underscore; defined as `_MODELS_DIRECT` on line 61.
Causes `NameError` on auto-download, blocking all embedding operations.
**Fix:** rename to `_MODELS_DIRECT` on line 35.

### BUG-2: Default model `nomic-ai/nomic-embed-v2` does not exist
**File:** `quipu/models/cache.py:12`
HF repo `nomic-ai/nomic-embed-v2` returns 404. `nomic-embed-text-v1.5` works.
**Impact:** All operations fail out-of-box. User must manually set `QUIPU_EMBEDDING_MODEL`.
**Fix:** change default to `nomic-embed-text-v1.5` or another verified-existing model.

## HIGH

### BUG-3: `cmd_receipts` hangs on non-interactive Windows
**File:** `quipu/cli.py:253` → `quipu/keystore/_backend.py:256`
`getpass.getpass("Quipu passphrase: ")` blocks forever when no TTY (CI, subprocess).
No `sys.stdin.isatty()` guard.
**Fix:** check `sys.stdin.isatty()` before calling `getpass`; raise clear error instead of hanging.

## MEDIUM

### BUG-4: Deprecated `huggingface_hub` arguments
**File:** `quipu/models/loader.py:42-46`
- `resume_download=True` — deprecated, ignored
- `local_dir_use_symlinks=False` — deprecated, ignored
**Fix:** remove both kwargs; `snapshot_download` handles resume and symlinks by default.

### BUG-5: No graceful fallback when embedding model unavailable
**File:** `quipu/retrieval/_search.py`
`search(R1)` and `search(R3)` crash with `ModelNotFoundError` instead of degrading to R2 (BM25-only).
**Fix:** catch `ModelNotFoundError` in search tiers; fall back to BM25 with a warning.

## NOTED (design limitations, not bugs)

### N-1: `store.insert()` produces atoms with no embedding
Low-level API doesn't trigger embedding pipeline. Only `write()` from `pipeline.py` embeds.
This is by design — `insert()` is raw storage.

### N-2: `write()` returns `str` (atom ID), not `dict`
API design choice. Caller uses `store.get(result)` to get full atom data.

### N-3: R1 can produce negative cosine scores
Min-max normalization in RRF fusion handles this; not functionally broken.

### N-4: `quipu` command not on PATH after `pip install`
Standard pip behavior on Windows. Documented workaround: `python -m quipu`.
