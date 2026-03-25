import os
from typing import Iterator, List, Optional


SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx"}


SKIP_DIRS = {
    "__pycache__", ".git", ".svn", "node_modules",
    ".venv", "venv", ".env", "env",
    "dist", "build", ".idea", ".vscode",
}


def crawl(
    root: str,
    extensions: Optional[set] = None,
    max_file_size_mb: float = 50.0,
) -> Iterator[str]:
    """
    Recursively walk `root` and yield absolute paths to supported files.

    Parameters
    ----------
    root             : Directory to start crawling from.
    extensions       : Override the default supported extension set.
    max_file_size_mb : Skip files larger than this (avoids memory issues).
    """
    allowed = extensions or SUPPORTED_EXTENSIONS
    max_bytes = max_file_size_mb * 1024 * 1024

    if not os.path.isdir(root):
        raise ValueError(f"Crawl root is not a directory: {root!r}")

    for dirpath, dirnames, filenames in os.walk(root, topdown=True):
        # Prune skip dirs in-place so os.walk won't descend into them
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]

        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in allowed:
                continue

            full_path = os.path.abspath(os.path.join(dirpath, fname))

            try:
                size = os.path.getsize(full_path)
            except OSError:
                continue

            if size > max_bytes:
                continue

            yield full_path


def crawl_to_list(root: str, **kwargs) -> List[str]:
    """Convenience wrapper that collects all results into a list."""
    return list(crawl(root, **kwargs))


def get_file_stats(paths: List[str]) -> dict:
    """Return a summary dict for a list of file paths."""
    by_ext: dict = {}
    total_bytes = 0

    for p in paths:
        ext = os.path.splitext(p)[1].lower()
        by_ext[ext] = by_ext.get(ext, 0) + 1
        try:
            total_bytes += os.path.getsize(p)
        except OSError:
            pass

    return {
        "total_files": len(paths),
        "by_extension": by_ext,
        "total_size_mb": round(total_bytes / (1024 * 1024), 2),
    }