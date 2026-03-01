"""CLI entry point for linmem."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from . import __version__
from .config import LinmemConfig


def _get_config(args) -> LinmemConfig:
    """Load or create config based on CLI args."""
    data_dir = getattr(args, "data_dir", ".linmem")
    config_path = Path(data_dir) / "config.json"
    if config_path.exists():
        cfg = LinmemConfig.load(config_path)
    else:
        cfg = LinmemConfig(data_dir=data_dir)
    # override from CLI
    if hasattr(args, "language") and args.language:
        cfg.language = args.language
    if hasattr(args, "top_k") and args.top_k:
        cfg.retrieval_top_k = args.top_k
    return cfg


def cmd_index(args) -> None:
    """Index documents from a directory."""
    from .ingest import ingest_directory
    from .retriever import Retriever

    cfg = _get_config(args)
    directory = Path(args.directory)
    if not directory.is_dir():
        print(f"Error: {directory} is not a directory", file=sys.stderr)
        sys.exit(1)

    print(f"Indexing {directory} ...")
    chunks = ingest_directory(directory, cfg)
    if not chunks:
        print("No documents found.")
        return

    print(f"Found {len(chunks)} chunks, building index ...")
    retriever = Retriever(cfg)
    new_count = retriever.index_chunks(chunks)
    cfg.save()
    print(f"Done. {new_count} new chunks indexed.")


def cmd_search(args) -> None:
    """Search indexed documents."""
    from .retriever import Retriever

    cfg = _get_config(args)
    retriever = Retriever(cfg)
    results = retriever.search(args.query, top_k=cfg.retrieval_top_k)

    if not results:
        print("No results found.")
        return

    for i, r in enumerate(results, 1):
        print(f"\n--- Result {i} (score: {r['score']}) ---")
        print(r["text"][:500])


def cmd_ask(args) -> None:
    """Search + LLM answer generation."""
    from .llm import call_llm
    from .retriever import Retriever

    cfg = _get_config(args)
    retriever = Retriever(cfg)
    results = retriever.search(args.question, top_k=cfg.retrieval_top_k)

    if not results:
        print("No relevant documents found.")
        return

    passages = [r["text"] for r in results]
    print("Generating answer ...")
    answer = call_llm(args.question, passages, cfg)
    print(f"\n{answer}")
    print("\n--- Sources ---")
    for i, r in enumerate(results, 1):
        print(f"  [{i}] (score: {r['score']}) {r['text'][:100]}...")


def cmd_status(args) -> None:
    """Show index status."""
    from .retriever import Retriever

    cfg = _get_config(args)
    config_path = cfg.index_dir / "config.json"
    if not config_path.exists():
        print("No index found. Run 'linmem index <directory>' first.")
        return

    retriever = Retriever(cfg)
    status = retriever.status()
    print(f"Index: {cfg.data_dir}")
    print(f"Language: {cfg.language}")
    for k, v in status.items():
        print(f"  {k}: {v}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="linmem",
        description="Local memory search CLI (BM25 + LinearRAG)",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--data-dir", default=".linmem", help="Index data directory")
    parser.add_argument("--language", choices=["zh", "en"], help="Language override")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    sub = parser.add_subparsers(dest="command")

    # index
    p_index = sub.add_parser("index", help="Index documents from a directory")
    p_index.add_argument("directory", help="Path to document directory")
    p_index.set_defaults(func=cmd_index)

    # search
    p_search = sub.add_parser("search", help="Search indexed documents")
    p_search.add_argument("query", help="Search query")
    p_search.add_argument("-k", "--top-k", type=int, help="Number of results")
    p_search.set_defaults(func=cmd_search)

    # ask
    p_ask = sub.add_parser("ask", help="Ask a question (search + LLM)")
    p_ask.add_argument("question", help="Your question")
    p_ask.add_argument("-k", "--top-k", type=int, help="Number of context passages")
    p_ask.set_defaults(func=cmd_ask)

    # status
    p_status = sub.add_parser("status", help="Show index status")
    p_status.set_defaults(func=cmd_status)

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(name)s %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING)

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
