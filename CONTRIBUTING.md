# Contributing to Quipu

Thanks for your interest in Quipu. Contributions of all sizes are welcome —
bug reports, docs fixes, and code.

## Good first issues

New here? Look for issues labelled **`good first issue`** — they are scoped to
be approachable without deep knowledge of the codebase.

## Getting started

```sh
git clone https://github.com/ajadi/quipu && cd quipu
pip install -e ".[vec]"
python -m pytest -q
```

## Workflow

1. Open (or comment on) an issue describing the change before large work.
2. Create a branch off `main`.
3. Keep changes focused and minimal — one logical change per pull request.
4. Add or update tests for new behaviour; run `python -m pytest -q` locally.
5. Use clear, imperative commit messages ("add X", "fix Y").
6. Open a pull request describing what changed and why.

## Guidelines

- **Local-first by default.** Core memory paths (write, search, capture,
  embeddings, retrieval) must run fully on-device with no network calls.
  Network access is opt-in only and must degrade gracefully when absent.
- **No new heavy dependencies** without discussion — Quipu is deliberately
  single-file and zero-infra.
- **Secrets via environment variables only** — never commit keys or tokens.

## Reporting bugs

Open an issue with your OS/Python version, the command you ran, and the full
output. Minimal reproductions are hugely appreciated.

By contributing, you agree that your contributions are licensed under the
[MIT License](LICENSE).
