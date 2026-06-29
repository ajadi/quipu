"""Entry point for ``python -m quipu``.

Usage::

    python -m quipu mirror --project-id <id> [--output-dir memory] [--db-path /path/to.db]
    python -m quipu serve
    python -m quipu init [--mode project|global|server]
    python -m quipu --version
"""

import argparse
import sys

from quipu.cli import cmd_backfill, cmd_drain, cmd_gc, cmd_init, cmd_receipts, get_version


def _cmd_mirror(args: argparse.Namespace) -> None:
    from quipu.storage import store as _store_factory
    from quipu.mirror import render_to_md

    s = _store_factory(args.db_path or None)
    try:
        result = render_to_md(args.project_id, args.output_dir, store=s)
    finally:
        s.close()

    total_atoms = sum(result.values())
    total_files = len(result)
    print(f"mirrored {total_atoms} atoms across {total_files} files")


def _cmd_serve(args: argparse.Namespace) -> None:
    import asyncio
    from quipu.server import run_stdio

    asyncio.run(run_stdio())


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="quipu",
        description="Quipu local-first memory system.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"quipu {get_version()}",
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    mirror_p = sub.add_parser("mirror", help="Render Quipu records to memory/*.md files.")
    mirror_p.add_argument("--project-id", required=True, help="Quipu project identifier.")
    mirror_p.add_argument(
        "--output-dir",
        default="memory",
        help="Output directory for .md files (default: memory).",
    )
    mirror_p.add_argument(
        "--db-path",
        default=None,
        help="Path to the Quipu SQLite DB (default: resolved via QUIPU_DB_PATH or ~/.quipu/quipu.db).",
    )

    sub.add_parser("serve", help="Start the Quipu MCP server over stdio.")

    init_p = sub.add_parser("init", help="Initialise the Quipu store and write config.json.")
    init_p.add_argument(
        "--mode",
        choices=["project", "global", "server"],
        default=None,
        help="Initialisation mode (default: project).",
    )

    drain_p = sub.add_parser("drain", help="Drain the capture queue and write records to the store.")
    drain_p.add_argument(
        "--queue-path",
        default=None,
        dest="queue_path",
        help="Path to capture-queue.jsonl (default: <project-root>/.quipu/capture-queue.jsonl).",
    )
    drain_p.add_argument(
        "--db-path",
        default=None,
        dest="db_path",
        help="Path to the Quipu SQLite DB (default: resolved via QUIPU_DB_PATH or mode).",
    )
    drain_p.add_argument(
        "--project-id",
        default=None,
        dest="project_id",
        help="Bound project scope; only records with this project_id are written.",
    )

    backfill_p = sub.add_parser(
        "backfill",
        help="Re-emit pre-existing atoms into the oplog so they sync to the hub.",
    )
    backfill_p.add_argument(
        "--db-path",
        default=None,
        dest="db_path",
        help="Path to the Quipu SQLite DB (default: resolved via QUIPU_DB_PATH or mode).",
    )
    backfill_p.add_argument(
        "--project-id",
        default=None,
        dest="project_id",
        help="Project scope to backfill (default: derived from the project root).",
    )

    receipts_p = sub.add_parser(
        "receipts",
        help="Export a hashed/redacted operation log for audit without plaintext content.",
    )
    receipts_p.add_argument(
        "--db-path",
        default=None,
        dest="db_path",
        help="Path to the Quipu SQLite DB (default: resolved via QUIPU_DB_PATH or mode).",
    )
    receipts_p.add_argument(
        "--project-id",
        default=None,
        dest="project_id",
        help="Project identifier (default: derived from the project root).",
    )
    receipts_p.add_argument(
        "--limit",
        type=int,
        default=None,
        dest="limit",
        help="Maximum receipt entries to return.",
    )
    receipts_p.add_argument(
        "--format",
        choices=["json", "text"],
        default="json",
        dest="fmt",
        help="Output format (default: json).",
    )
    receipts_p.add_argument(
        "--op",
        choices=["write", "invalidate"],
        default=None,
        dest="op_filter",
        help="Filter by operation type (default: all).",
    )

    gc_p = sub.add_parser(
        "gc",
        help="Garbage collect stale low-value atoms (opt-in, reversible).",
    )
    gc_p.add_argument(
        "--db-path",
        default=None,
        dest="db_path",
        help="Path to the Quipu SQLite DB (default: resolved via QUIPU_DB_PATH or mode).",
    )
    gc_p.add_argument(
        "--project-id",
        default=None,
        dest="project_id",
        help="Project identifier (default: from config or QUIPU_PROJECT_ID).",
    )
    gc_p.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        dest="dry_run",
        help="List stale candidates without taking action (default).",
    )
    gc_p.add_argument(
        "--run",
        action="store_true",
        default=False,
        dest="run_flag",
        help="Soft-invalidate stale atoms (reversible).",
    )
    gc_p.add_argument(
        "--min-age-days",
        type=int,
        default=90,
        dest="min_age_days",
        help="Minimum age in days to consider stale (default 90).",
    )
    gc_p.add_argument(
        "--min-access-count",
        type=int,
        default=3,
        dest="min_access_count",
        help="Atoms with access_count below this are stale (default 3).",
    )

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "mirror":
        _cmd_mirror(args)
    elif args.command == "serve":
        _cmd_serve(args)
    elif args.command == "init":
        sys.exit(cmd_init(args.mode))
    elif args.command == "drain":
        sys.exit(cmd_drain(args.queue_path, args.db_path, args.project_id))
    elif args.command == "backfill":
        sys.exit(cmd_backfill(args.db_path, args.project_id))
    elif args.command == "receipts":
        sys.exit(cmd_receipts(args.db_path, args.project_id, args.limit, args.fmt, args.op_filter))
    elif args.command == "gc":
        sys.exit(cmd_gc(args.db_path, args.project_id, args.dry_run, args.run_flag, args.min_age_days, args.min_access_count))


if __name__ == "__main__":
    main()
