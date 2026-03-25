import math
from typing import List, Dict

from database import db


def index_file(file_id: int, term_data: Dict[str, Dict]):
    """
    Persist term data for a single file.

    Parameters
    ----------
    file_id   : Primary key from the files table.
    term_data : Output of preprocessor.compute_term_data().
    """
    db.insert_index_entries(file_id, term_data)


def compute_and_store_tfidf():
    """
    Calculate TF-IDF for every (term, document) pair in the index
    and persist the results.

    TF-IDF formula used:
      tf-idf(t, d) = tf(t, d) × log( (N + 1) / (df(t) + 1) ) + 1
      (smooth IDF variant to avoid division-by-zero on unseen terms)
    """
    N = db.get_total_docs()
    if N == 0:
        return

    vocab = db.get_all_vocabulary()
    scores: List[tuple] = []

    for term in vocab:
        df      = db.get_doc_frequency(term)
        idf     = math.log((N + 1) / (df + 1)) + 1
        postings = db.get_posting_list(term)

        for posting in postings:
            tfidf = posting["tf"] * idf
            scores.append((term, posting["file_id"], tfidf))

    db.insert_tfidf_scores(scores)


def get_postings(term: str) -> List[Dict]:
    """Return posting list for a single term (used by the search module)."""
    return db.get_posting_list(term)


def get_tfidf_ranking(term: str) -> List[Dict]:
    """Return files sorted by TF-IDF score for a term."""
    return db.get_tfidf_scores_for_term(term)