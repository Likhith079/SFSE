import os
import sys
import threading
import time
from flask import Flask, request, jsonify, render_template, abort

# Add project root to path
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
from parser.parser import parse_file

app = Flask(__name__, template_folder="templates", static_folder="static")

# ── Global indexing state (for progress reporting) ──────────────────────────

_index_state = {
    "running":  False,
    "progress": "",
    "done":     False,
    "error":    "",
    "summary":  None,
}


def _run_index_thread(root_dir: str, reindex: bool):
    global _index_state
    _index_state.update(running=True, done=False, error="", summary=None, progress="Starting…")
    try:
        summary = run_indexing(root_dir, verbose=False, reindex=reindex)
        _index_state.update(
            running=False, done=True,
            progress="Indexing complete.",
            summary=summary,
        )
    except Exception as exc:
        _index_state.update(running=False, done=True, error=str(exc), progress="Failed.")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/index", methods=["POST"])
def api_index():
    data     = request.get_json(silent=True) or {}
    root_dir = data.get("directory", "").strip()
    reindex  = bool(data.get("reindex", False))

    if not root_dir:
        return jsonify({"error": "directory is required"}), 400
    if not os.path.isdir(root_dir):
        return jsonify({"error": f"Not a valid directory: {root_dir}"}), 400
    if _index_state["running"]:
        return jsonify({"error": "Indexing already in progress"}), 409

    t = threading.Thread(target=_run_index_thread, args=(root_dir, reindex), daemon=True)
    t.start()
    return jsonify({"status": "started", "directory": root_dir})


@app.route("/api/index/status")
def api_index_status():
    return jsonify(_index_state)


@app.route("/api/index", methods=["DELETE"])
def api_clear_index():
    db.initialize_db()
    db.clear_index()
    _index_state.update(running=False, done=False, progress="", summary=None, error="")
    return jsonify({"status": "cleared"})


@app.route("/api/search")
def api_search():
    query    = request.args.get("q", "").strip()
    strategy = request.args.get("strategy", "tfidf")
    top_k    = int(request.args.get("top_k", 10))

    if not query:
        return jsonify({"error": "q parameter is required"}), 400

    db.initialize_db()
    t0 = time.time()

    valid_strategies = {"keyword", "tfidf", "fuzzy", "semantic", "all"}
    if strategy not in valid_strategies:
        strategy = "tfidf"

    results = []

    if strategy == "keyword":
        results = keyword_search(query)[:top_k]
    elif strategy == "tfidf":
        results = tfidf_search(query, top_k=top_k)
    elif strategy == "fuzzy":
        results = fuzzy_search(query, top_k=top_k)
    elif strategy == "semantic":
        results = semantic_search(query, top_k=top_k)
    elif strategy == "all":
        kw  = keyword_search(query)[:top_k]
        tf  = tfidf_search(query,  top_k=top_k)
        fz  = fuzzy_search(query,  top_k=top_k)
        sem = semantic_search(query, top_k=top_k)
        results = rank_results(kw, tf, fz, sem)[:top_k]

    elapsed_ms = round((time.time() - t0) * 1000, 1)

    # Sanitise for JSON (ensure all fields present)
    clean = []
    for r in results:
        clean.append({
            "file_id":  r.get("file_id"),
            "filename": r.get("filename", ""),
            "path":     r.get("path", ""),
            "score":    round(float(r.get("score", 0)), 5),
            "strategy": r.get("strategy", strategy),
            "ext":      os.path.splitext(r.get("filename", ""))[1].lower(),
        })

    return jsonify({
        "query":      query,
        "strategy":   strategy,
        "results":    clean,
        "count":      len(clean),
        "elapsed_ms": elapsed_ms,
    })


@app.route("/api/stats")
def api_stats():
    db.initialize_db()
    files = db.get_all_files()
    vocab = db.get_all_vocabulary()

    by_ext: dict = {}
    for f in files:
        ext = f.get("extension", "unknown")
        by_ext[ext] = by_ext.get(ext, 0) + 1

    return jsonify({
        "total_files": len(files),
        "vocabulary":  len(vocab),
        "by_extension": by_ext,
    })


@app.route("/api/preview/<int:file_id>")
def api_preview(file_id: int):
    db.initialize_db()
    all_files = {f["id"]: f for f in db.get_all_files()}
    f = all_files.get(file_id)
    if not f:
        abort(404)

    try:
        text = parse_file(f["path"])
        words = text.split()
        snippet = " ".join(words[:400])
        if len(words) > 400:
            snippet += " …"
    except Exception as exc:
        snippet = f"[Preview error: {exc}]"

    return jsonify({
        "file_id":  file_id,
        "filename": f["filename"],
        "path":     f["path"],
        "preview":  snippet,
    })


if __name__ == "__main__":
    db.initialize_db()
    port = int(os.environ.get("PORT", 5000))
    print(f"\n Smart File Search Engine  →  http://localhost:{port}\n")
    app.run(debug=True, port=port)