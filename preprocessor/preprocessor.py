import re
import string
from typing import List, Dict, Optional


# ── NLTK lazy initialisation ───────────────────────────────────────────────────

_stop_words: Optional[set] = None
_stemmer = None


def _get_stopwords() -> set:
    global _stop_words
    if _stop_words is None:
        try:
            from nltk.corpus import stopwords
            import nltk
            try:
                _stop_words = set(stopwords.words("english"))
            except LookupError:
                nltk.download("stopwords", quiet=True)
                _stop_words = set(stopwords.words("english"))
        except ImportError:
            # Fallback minimal stop-word list if NLTK unavailable
            _stop_words = {
                "a", "an", "the", "is", "it", "in", "on", "at", "to",
                "for", "of", "and", "or", "but", "not", "with", "this",
                "that", "was", "are", "be", "as", "by", "from", "its",
            }
    return _stop_words


def _get_stemmer():
    global _stemmer
    if _stemmer is None:
        try:
            from nltk.stem import PorterStemmer
            import nltk
            _stemmer = PorterStemmer()
        except ImportError:
            _stemmer = _NullStemmer()
    return _stemmer


class _NullStemmer:
    """No-op stemmer when NLTK is unavailable."""
    def stem(self, word: str) -> str:
        return word


# ── Public API ─────────────────────────────────────────────────────────────────

def preprocess(text: str) -> List[str]:
    """
    Return a list of cleaned, stemmed tokens from raw text.
    Order is preserved; duplicates are kept (needed for TF computation).
    """
    if not text:
        return []

    # 1. Lowercase
    text = text.lower()

    # 2. Remove non-alphanumeric characters (keep spaces)
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # 3. Tokenise
    tokens = text.split()

    # 4. Remove stop words and very short tokens
    stop_words = _get_stopwords()
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]

    # 5. Stem
    stemmer = _get_stemmer()
    tokens = [stemmer.stem(t) for t in tokens]

    return tokens


def compute_term_data(tokens: List[str]) -> Dict[str, Dict]:
    """
    Given an ordered token list, return:
    {
      term: {
        "tf":        float,    # term frequency (normalised)
        "positions": [int, ...]  # token-position offsets
      },
      ...
    }
    """
    if not tokens:
        return {}

    total = len(tokens)
    term_data: Dict[str, Dict] = {}

    for pos, token in enumerate(tokens):
        if token not in term_data:
            term_data[token] = {"count": 0, "positions": []}
        term_data[token]["count"] += 1
        term_data[token]["positions"].append(pos)

    # Normalise TF
    for data in term_data.values():
        data["tf"] = data["count"] / total
        del data["count"]

    return term_data


def preprocess_query(query: str) -> List[str]:
    """
    Preprocess a search query.
    Same pipeline as document tokens but returns unique terms only.
    """
    tokens = preprocess(query)
    seen = set()
    unique = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique