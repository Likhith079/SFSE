import argparse
import os
import sys
import time

# Add project root to path when run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pipeline import run_indexing
from search.search_engine import (
    keyword_search,
    tfidf_search,
    fuzzy_search,
    semantic_search,
)
from search.ranking import rank_results
from database import db


# ── Helpers ───────────────────────────────────────────────────────────────────

def _print_results(results, strategy_label):
    if not results:
        print(f"  No results from [{strategy_label}]")
        return
    print(f"\n  Results [{strategy_label}] — {len(results)} match(es):")
    print(f"  {'#':<4} {'Score':<10} {'Filename':<40} Path")
    print("  " + "-" * 100)
    for i, r in enumerate(results, start=1):
        print(f"  {i:<4} {r['score']:<10.5f} {r['filename']:<40} {r['path']}")


def _try_rich():
    """Return a Rich Console if available, else None."""
    try:
        from rich.console import Console
        from rich.table   import Table
        return Console(), Table
    except ImportError:
        return None, None


# ── Sub-commands ──────────────────────────────────────────────────────────────

def cmd_index(args):
    root = os.path.abspath(args.directory)
    print(f"\nIndexing directory: {root}\n")
    t0 = time.time()
    summary = run_indexing(root, verbose=True, reindex=args.reindex)
    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.2f}s — {summary}")


def cmd_search(args):
    query    = args.query
    strategy = args.strategy
    top_k    = args.top_k

    print(f'\nSearching for: "{query}"  [strategy={strategy}  top_k={top_k}]\n')
    t0 = time.time()

    db.initialize_db()

    if strategy in ("keyword", "all"):
        res = keyword_search(query)
        _print_results(res[:top_k], "keyword")

    if strategy in ("tfidf", "all"):
        res = tfidf_search(query, top_k=top_k)
        _print_results(res, "tfidf")

    if strategy in ("fuzzy", "all"):
        res = fuzzy_search(query, top_k=top_k)
        _print_results(res, "fuzzy")

    if strategy in ("semantic", "all"):
        res = semantic_search(query, top_k=top_k)
        _print_results(res, "semantic")

    if strategy == "all":
        # Fused ranking
        kw  = keyword_search(query)[:top_k]
        tf  = tfidf_search(query,  top_k=top_k)
        fz  = fuzzy_search(query,  top_k=top_k)
        sem = semantic_search(query, top_k=top_k)
        fused = rank_results(kw, tf, fz, sem)
        _print_results(fused[:top_k], "FUSED (RRF)")

    elapsed = time.time() - t0
    print(f"\nSearch completed in {elapsed:.3f}s\n")


def cmd_stats(args):
    db.initialize_db()
    files = db.get_all_files()
    vocab = db.get_all_vocabulary()

    print(f"\n{'─'*50}")
    print(f"  Indexed files : {len(files)}")
    print(f"  Vocabulary    : {len(vocab)} unique terms")
    if files:
        by_ext: dict = {}
        for f in files:
            by_ext[f['extension']] = by_ext.get(f['extension'], 0) + 1
        print(f"  By extension  : {by_ext}")
    print(f"{'─'*50}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="smart-search",
        description="Smart File System Search Engine",
    )
    sub = parser.add_subparsers(dest="command")

    # index sub-command
    p_index = sub.add_parser("index", help="Index a directory")
    p_index.add_argument("directory", help="Root directory to index")
    p_index.add_argument("--reindex", action="store_true",
                         help="Clear existing index before indexing")

    # search sub-command
    p_search = sub.add_parser("search", help="Search the index")
    p_search.add_argument("query", help="Search query string")
    p_search.add_argument(
        "--strategy",
        choices=["keyword", "tfidf", "fuzzy", "semantic", "all"],
        default="tfidf",
        help="Retrieval strategy (default: tfidf)",
    )
    p_search.add_argument(
        "--top-k", type=int, default=10,
        dest="top_k",
        help="Number of results to return (default: 10)",
    )

    # stats sub-command
    sub.add_parser("stats", help="Show index statistics")

    args = parser.parse_args()

    if args.command == "index":
        cmd_index(args)
    elif args.command == "search":
        cmd_search(args)
    elif args.command == "stats":
        cmd_stats(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()