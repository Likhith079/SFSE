import sqlite3
import json
import os
from typing import List, Dict, Tuple, Optional


DB_PATH = os.path.join(os.path.dirname(__file__), "..", "search_index.db")


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_db():
    """Create all required tables if they don't exist."""
    conn = get_connection()
    cur = conn.cursor()

    cur.executescript("""
        CREATE TABLE IF NOT EXISTS files (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            path        TEXT    UNIQUE NOT NULL,
            filename    TEXT    NOT NULL,
            extension   TEXT    NOT NULL,
            size_bytes  INTEGER,
            indexed_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS inverted_index (
            term        TEXT    NOT NULL,
            file_id     INTEGER NOT NULL,
            tf          REAL    DEFAULT 0,
            positions   TEXT,           -- JSON list of character offsets
            PRIMARY KEY (term, file_id),
            FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS tfidf_scores (
            term        TEXT    NOT NULL,
            file_id     INTEGER NOT NULL,
            score       REAL    NOT NULL,
            PRIMARY KEY (term, file_id),
            FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_term      ON inverted_index(term);
        CREATE INDEX IF NOT EXISTS idx_file      ON inverted_index(file_id);
        CREATE INDEX IF NOT EXISTS idx_tfidf_term ON tfidf_scores(term);
    """)

    conn.commit()
    conn.close()


# ── File metadata ──────────────────────────────────────────────────────────────

def upsert_file(path: str) -> int:
    """Insert or update a file record, return its id."""
    filename  = os.path.basename(path)
    extension = os.path.splitext(filename)[1].lower()
    size      = os.path.getsize(path) if os.path.exists(path) else 0

    conn = get_connection()
    cur  = conn.cursor()
    cur.execute("""
        INSERT INTO files (path, filename, extension, size_bytes)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(path) DO UPDATE SET
            filename   = excluded.filename,
            extension  = excluded.extension,
            size_bytes = excluded.size_bytes,
            indexed_at = CURRENT_TIMESTAMP
    """, (path, filename, extension, size))
    conn.commit()

    cur.execute("SELECT id FROM files WHERE path = ?", (path,))
    file_id = cur.fetchone()["id"]
    conn.close()
    return file_id


def get_file_id(path: str) -> Optional[int]:
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute("SELECT id FROM files WHERE path = ?", (path,))
    row = cur.fetchone()
    conn.close()
    return row["id"] if row else None


def get_all_files() -> List[Dict]:
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute("SELECT * FROM files")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def get_total_docs() -> int:
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute("SELECT COUNT(*) AS n FROM files")
    n = cur.fetchone()["n"]
    conn.close()
    return n


# ── Inverted index ─────────────────────────────────────────────────────────────

def insert_index_entries(file_id: int, term_data: Dict[str, Dict]):
    """
    term_data: { term: { "tf": float, "positions": [int, ...] } }
    """
    conn = get_connection()
    cur  = conn.cursor()

    # Delete stale entries for this file before re-inserting
    cur.execute("DELETE FROM inverted_index WHERE file_id = ?", (file_id,))

    rows = [
        (term, file_id, data["tf"], json.dumps(data.get("positions", [])))
        for term, data in term_data.items()
    ]
    cur.executemany("""
        INSERT INTO inverted_index (term, file_id, tf, positions)
        VALUES (?, ?, ?, ?)
    """, rows)
    conn.commit()
    conn.close()


def get_posting_list(term: str) -> List[Dict]:
    """Return [{file_id, tf, path, filename}, ...] for a given term."""
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute("""
        SELECT ii.file_id, ii.tf, f.path, f.filename
        FROM   inverted_index ii
        JOIN   files f ON f.id = ii.file_id
        WHERE  ii.term = ?
    """, (term,))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def get_doc_frequency(term: str) -> int:
    """Number of documents containing this term."""
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute("SELECT COUNT(*) AS n FROM inverted_index WHERE term = ?", (term,))
    n = cur.fetchone()["n"]
    conn.close()
    return n


def get_all_terms_for_file(file_id: int) -> List[str]:
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute("SELECT term FROM inverted_index WHERE file_id = ?", (file_id,))
    terms = [r["term"] for r in cur.fetchall()]
    conn.close()
    return terms


# ── TF-IDF scores ──────────────────────────────────────────────────────────────

def insert_tfidf_scores(scores: List[Tuple[str, int, float]]):
    """scores: [(term, file_id, score), ...]"""
    conn = get_connection()
    cur  = conn.cursor()
    cur.executemany("""
        INSERT INTO tfidf_scores (term, file_id, score)
        VALUES (?, ?, ?)
        ON CONFLICT(term, file_id) DO UPDATE SET score = excluded.score
    """, scores)
    conn.commit()
    conn.close()


def get_tfidf_scores_for_term(term: str) -> List[Dict]:
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute("""
        SELECT ts.file_id, ts.score, f.path, f.filename
        FROM   tfidf_scores ts
        JOIN   files f ON f.id = ts.file_id
        WHERE  ts.term = ?
        ORDER  BY ts.score DESC
    """, (term,))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def get_all_vocabulary() -> List[str]:
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute("SELECT DISTINCT term FROM inverted_index")
    vocab = [r["term"] for r in cur.fetchall()]
    conn.close()
    return vocab


def clear_index():
    conn = get_connection()
    cur  = conn.cursor()
    cur.executescript("""
        DELETE FROM tfidf_scores;
        DELETE FROM inverted_index;
        DELETE FROM files;
    """)
    conn.commit()
    conn.close()