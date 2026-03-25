import math
from collections import defaultdict
from typing import List, Dict, Optional

from preprocessor.preprocessor import preprocess_query
from database import db


# ── 1. Keyword search ──────────────────────────────────────────────────────────

def keyword_search(query: str) -> List[Dict]:
    """
    Boolean AND search: return files that contain ALL query terms.
    Results are sorted by combined TF score (descending).
    """
    terms = preprocess_query(query)
    if not terms:
        return []

    # For each term get the set of matching file_ids
    posting_sets = []
    tf_accum: Dict[int, float] = defaultdict(float)

    for term in terms:
        postings = db.get_posting_list(term)
        ids = set()
        for p in postings:
            ids.add(p["file_id"])
            tf_accum[p["file_id"]] += p["tf"]
        posting_sets.append(ids)

    # Intersect: keep only files that match ALL terms
    if not posting_sets:
        return []
    common_ids = posting_sets[0]
    for s in posting_sets[1:]:
        common_ids &= s

    if not common_ids:
        return []

    all_files = {f["id"]: f for f in db.get_all_files()}
    results = []
    for fid in common_ids:
        f = all_files.get(fid, {})
        results.append({
            "file_id":  fid,
            "path":     f.get("path", ""),
            "filename": f.get("filename", ""),
            "score":    round(tf_accum[fid], 6),
            "strategy": "keyword",
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


# ── 2. TF-IDF search ──────────────────────────────────────────────────────────

def tfidf_search(query: str, top_k: int = 10) -> List[Dict]:
    """
    Rank documents by summed TF-IDF score across all query terms.
    Falls back to keyword posting if TF-IDF scores haven't been computed yet.
    """
    terms = preprocess_query(query)
    if not terms:
        return []

    score_accum: Dict[int, float] = defaultdict(float)
    path_map:    Dict[int, Dict]  = {}

    for term in terms:
        rows = db.get_tfidf_scores_for_term(term)
        if not rows:
            # Fallback: use raw TF
            rows = db.get_posting_list(term)
            for r in rows:
                score_accum[r["file_id"]] += r.get("tf", 0)
                path_map[r["file_id"]] = r
        else:
            for r in rows:
                score_accum[r["file_id"]] += r["score"]
                path_map[r["file_id"]] = r

    if not score_accum:
        return []

    results = []
    for fid, score in score_accum.items():
        info = path_map.get(fid, {})
        results.append({
            "file_id":  fid,
            "path":     info.get("path", ""),
            "filename": info.get("filename", ""),
            "score":    round(score, 6),
            "strategy": "tfidf",
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


# ── 3. Fuzzy search ────────────────────────────────────────────────────────────

def fuzzy_search(query: str, threshold: int = 70, top_k: int = 10) -> List[Dict]:
    """
    Find documents whose vocabulary contains terms similar to the query terms.
    Uses fuzzywuzzy's token_set_ratio for comparison.

    Parameters
    ----------
    threshold : Minimum similarity score (0-100). Higher = stricter.
    """
    try:
        from fuzzywuzzy import fuzz
    except ImportError:
        print("[fuzzy] fuzzywuzzy not installed, falling back to keyword search")
        return keyword_search(query)

    query_terms = preprocess_query(query)
    if not query_terms:
        return []

    vocab = db.get_all_vocabulary()
    if not vocab:
        return []

    # Find similar vocabulary terms for each query term
    matched_terms = set()
    for qt in query_terms:
        for vt in vocab:
            ratio = fuzz.token_set_ratio(qt, vt)
            if ratio >= threshold:
                matched_terms.add(vt)

    if not matched_terms:
        return []

    # Aggregate TF scores for matched terms
    score_accum: Dict[int, float] = defaultdict(float)
    path_map:    Dict[int, Dict]  = {}

    for term in matched_terms:
        for posting in db.get_posting_list(term):
            score_accum[posting["file_id"]] += posting["tf"]
            path_map[posting["file_id"]] = posting

    results = []
    for fid, score in score_accum.items():
        info = path_map.get(fid, {})
        results.append({
            "file_id":  fid,
            "path":     info.get("path", ""),
            "filename": info.get("filename", ""),
            "score":    round(score, 6),
            "strategy": "fuzzy",
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


# ── 4. Semantic search ─────────────────────────────────────────────────────────

class SemanticSearchEngine:
    """
    Lazy-loaded sentence-transformer based semantic search.
    Embeddings are computed on-the-fly from stored file text.

    NOTE: For production use, pre-compute and cache embeddings in the DB.
    This implementation re-reads file content each query for simplicity.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._model      = None

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                print(f"[semantic] Loading model '{self._model_name}' …")
                self._model = SentenceTransformer(self._model_name)
                print("[semantic] Model ready.")
            except ImportError:
                raise RuntimeError(
                    "sentence-transformers not installed. "
                    "Run: pip install sentence-transformers"
                )

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Compute cosine similarity between query embedding and document embeddings.
        Document text is read from disk each call (see note above).
        """
        self._load_model()

        from sentence_transformers import util as st_util
        import torch

        from parser.parser import parse_file

        all_files = db.get_all_files()
        if not all_files:
            return []

        # Build corpus from file content (first 512 words to keep it fast)
        corpus_texts = []
        corpus_meta  = []
        for f in all_files:
            try:
                text = parse_file(f["path"])
                snippet = " ".join(text.split()[:512])
                corpus_texts.append(snippet if snippet else f["filename"])
                corpus_meta.append(f)
            except Exception:
                corpus_texts.append(f["filename"])
                corpus_meta.append(f)

        query_emb  = self._model.encode(query,        convert_to_tensor=True)
        corpus_emb = self._model.encode(corpus_texts, convert_to_tensor=True, show_progress_bar=False)

        cosine_scores = st_util.cos_sim(query_emb, corpus_emb)[0]
        top_results   = torch.topk(cosine_scores, k=min(top_k, len(corpus_texts)))

        results = []
        for score, idx in zip(top_results.values, top_results.indices):
            f = corpus_meta[idx]
            results.append({
                "file_id":  f["id"],
                "path":     f["path"],
                "filename": f["filename"],
                "score":    round(float(score), 4),
                "strategy": "semantic",
            })

        return results


# Module-level singleton so the model loads only once per session
_semantic_engine: Optional[SemanticSearchEngine] = None


def semantic_search(query: str, top_k: int = 5) -> List[Dict]:
    global _semantic_engine
    if _semantic_engine is None:
        _semantic_engine = SemanticSearchEngine()
    return _semantic_engine.search(query, top_k=top_k)