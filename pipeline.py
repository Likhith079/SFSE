import os
import sys
from typing import Optional
 
from crawler.crawler     import crawl, get_file_stats
from parser.parser       import parse_file
from preprocessor.preprocessor import preprocess, compute_term_data
from index.indexer       import index_file, compute_and_store_tfidf
from database            import db
 
 
def run_indexing(
    root_dir: str,
    verbose: bool = True,
    reindex: bool = False,
) -> dict:
    """
    Index all supported files under `root_dir`.
 
    Parameters
    ----------
    root_dir  : Path to directory to index.
    verbose   : Print progress to stdout.
    reindex   : If True, clear the existing index before starting.
 
    Returns
    -------
    Summary dict with counts and timing info.
    """
    if not os.path.isdir(root_dir):
        print(f"[pipeline] ERROR: {root_dir!r} is not a valid directory.")
        sys.exit(1)
 
    db.initialize_db()
 
    if reindex:
        if verbose:
            print("[pipeline] Clearing existing index …")
        db.clear_index()
 
    # ── Step 1: Crawl ──────────────────────────────────────────────────────────
    if verbose:
        print(f"[pipeline] Crawling {root_dir!r} …")
 
    file_paths = list(crawl(root_dir))
    stats = get_file_stats(file_paths)
 
    if verbose:
        print(f"[pipeline] Found {stats['total_files']} files  "
              f"({stats['total_size_mb']} MB)  "
              f"by extension: {stats['by_extension']}")
 
    indexed = 0
    skipped = 0
    errors  = 0
 
    # ── Steps 2–4: Parse → Preprocess → Index (per file) ──────────────────────
    for i, path in enumerate(file_paths, start=1):
        if verbose:
            print(f"  [{i}/{stats['total_files']}] {os.path.basename(path)}", end="  ")
 
        # Parse
        text = parse_file(path)
        if not text.strip():
            if verbose:
                print("SKIP (empty)")
            skipped += 1
            continue
 
        # Preprocess
        tokens = preprocess(text)
        if not tokens:
            if verbose:
                print("SKIP (no tokens)")
            skipped += 1
            continue
 
        term_data = compute_term_data(tokens)
 
        # Persist
        try:
            file_id = db.upsert_file(path)
            index_file(file_id, term_data)
            indexed += 1
            if verbose:
                print(f"OK  ({len(term_data)} unique terms)")
        except Exception as exc:
            errors += 1
            if verbose:
                print(f"ERROR: {exc}")
 
    # ── Step 5: Compute TF-IDF ────────────────────────────────────────────────
    if indexed > 0:
        if verbose:
            print("[pipeline] Computing TF-IDF scores …")
        compute_and_store_tfidf()
        if verbose:
            print("[pipeline] TF-IDF scores stored.")
 
    summary = {
        "total_found": stats["total_files"],
        "indexed":     indexed,
        "skipped":     skipped,
        "errors":      errors,
    }
 
    if verbose:
        print(f"\n[pipeline] Done. {indexed} indexed, "
              f"{skipped} skipped, {errors} errors.")
 
    return summary
 