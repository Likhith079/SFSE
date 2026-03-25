# SFSE
# Smart File System Search Engine

A local file search system that combines keyword retrieval, TF-IDF ranking, fuzzy matching, and AI-based semantic search into a single CLI tool.

---

## Project Structure

```
smart_search/
├── crawler/          # File system walker
├── parser/           # Text extractor (TXT, PDF, DOCX)
├── preprocessor/     # Tokeniser, stop-word removal, stemming
├── index/            # Inverted index builder + TF-IDF computation
├── database/         # SQLite persistence layer
├── search/           # Keyword, TF-IDF, fuzzy, semantic search engines
│   └── ranking.py    # Reciprocal Rank Fusion
├── ui/               # CLI interface
├── pipeline.py       # Full indexing pipeline orchestrator
└── requirements.txt
```

---

## Setup

### 1. Create a virtual environment 

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download NLTK data

```bash
python -c "import nltk; nltk.download('stopwords')"
```

---

## Usage

Run all commands from the `smart_search/` directory.

### Index a directory

```bash
python -m ui.cli index /path/to/your/documents
```

To clear and rebuild the index from scratch:

```bash
python -m ui.cli index /path/to/your/documents --reindex
```

### Search

```bash
# TF-IDF search (default)
python -m ui.cli search "machine learning algorithms"

# Keyword (boolean AND) search
python -m ui.cli search "neural network" --strategy keyword

# Fuzzy search (handles typos)
python -m ui.cli search "machin lerning" --strategy fuzzy

# Semantic search (requires sentence-transformers)
python -m ui.cli search "deep learning for images" --strategy semantic

# All strategies + Reciprocal Rank Fusion
python -m ui.cli search "data processing pipeline" --strategy all --top-k 5
```

### Show index statistics

```bash
python -m ui.cli stats
```

---

## Search Strategies

| Strategy | Description | Best for |
|----------|-------------|----------|
| `keyword` | Boolean AND on inverted index | Exact term matching |
| `tfidf`   | Ranked by TF-IDF score sum | General relevance ranking |
| `fuzzy`   | Levenshtein distance matching | Typo tolerance |
| `semantic`| Sentence-transformer cosine similarity | Conceptual/synonym matching |
| `all`     | All four + Reciprocal Rank Fusion | Best overall results |

---

## Supported File Formats

- `.txt` — Plain text
- `.pdf` — PDF documents (text-based, not scanned images)
- `.docx` — Microsoft Word documents

---

## Architecture

```
File System
    │
    ▼
 Crawler ──────► Parser ──────► Preprocessor ──────► Inverted Index
                                                           │
                                                           ▼
                                                        Database (SQLite)
                                                           │
                                                           ▼
Query ──► Keyword Search ─────────────────────────────► Ranking
      ──► TF-IDF Search  ─────────────────────────────►  (RRF)
      ──► Fuzzy Search   ─────────────────────────────►    │
      ──► Semantic Search ───────────────────────────────► ▼
                                                        Results


