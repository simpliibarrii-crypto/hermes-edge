"""
On-device RAG — local embeddings + SQLite FTS5 for knowledge-augmented generation.
No external vector DB required. Runs entirely offline.
"""
import json, logging, math, re, sqlite3, hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Tiny built-in tokenizer for chunking (no model dependency)
def _simple_tokenize(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", text.lower())

def _chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """Split text into overlapping chunks at sentence boundaries."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, current, current_len = [], [], 0
    for sent in sentences:
        sent_len = len(_simple_tokenize(sent))
        if current_len + sent_len > chunk_size and current:
            chunks.append(" ".join(current))
            # overlap: keep last sentences
            overlap_tokens = 0
            kept = []
            for s in reversed(current):
                t = len(_simple_tokenize(s))
                if overlap_tokens + t > overlap and kept:
                    break
                kept.insert(0, s)
                overlap_tokens += t
            current = kept
            current_len = overlap_tokens
        current.append(sent)
        current_len += sent_len
    if current:
        chunks.append(" ".join(current))
    return chunks

# Simple TF-IDF vector implementation (no numpy/scipy dependency)
def _compute_tfidf(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    tf = {}
    for t in tokens:
        tf[t] = tf.get(t, 0) + 1
    n = len(tokens)
    return {t: (tf[t] / n) * idf.get(t, 1.0) for t in tf}

def _cosine_sim(a: dict[str, float], b: dict[str, float]) -> float:
    keys = set(a) & set(b)
    if not keys:
        return 0.0
    dot = sum(a[k] * b[k] for k in keys)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0

@dataclass
class Document:
    id: str = ""
    title: str = ""
    content: str = ""
    source: str = ""
    metadata: dict = field(default_factory=dict)

@dataclass
class Chunk:
    doc_id: str
    text: str
    tokens: list[str]
    embedding: dict[str, float]  # sparse TF-IDF vector

class RAGEngine:
    """On-device RAG with SQLite-backed document store and TF-IDF retrieval."""

    def __init__(self, db_path: str = "~/.hermes/rag.db"):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._init_db()
        self._idf: dict[str, float] = {}
        self._chunks: list[Chunk] = []
        self._rebuild_index()

    def _init_db(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                title TEXT,
                content TEXT,
                source TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts
            USING fts5(title, content, tokenize=porter)
        """)
        self._conn.commit()

    def _rebuild_index(self):
        """Rebuild in-memory TF-IDF index from stored documents."""
        rows = self._conn.execute("SELECT id, content FROM documents").fetchall()
        all_tokens = []
        self._chunks = []
        for doc_id, content in rows:
            chunks = _chunk_text(content)
            for chunk_text in chunks:
                tokens = _simple_tokenize(chunk_text)
                all_tokens.extend(tokens)
                self._chunks.append(Chunk(doc_id, chunk_text, tokens, {}))
        # Compute IDF
        n_docs = len(self._chunks) or 1
        df = {}
        for c in self._chunks:
            for t in set(c.tokens):
                df[t] = df.get(t, 0) + 1
        self._idf = {t: math.log(n_docs / (v + 1)) + 1 for t, v in df.items()}
        # Compute embeddings
        for c in self._chunks:
            c.embedding = _compute_tfidf(c.tokens, self._idf)
        log.info("RAG index: %d chunks, %d terms", len(self._chunks), len(self._idf))

    def add_document(self, doc: Document) -> str:
        doc_id = doc.id or hashlib.sha256(doc.content.encode()).hexdigest()[:16]
        self._conn.execute(
            "INSERT OR REPLACE INTO documents (id, title, content, source, metadata) VALUES (?, ?, ?, ?, ?)",
            (doc_id, doc.title, doc.content, doc.source, json.dumps(doc.metadata)),
        )
        self._conn.execute(
            "INSERT OR REPLACE INTO docs_fts (rowid, title, content) VALUES ((SELECT rowid FROM documents WHERE id=?), ?, ?)",
            (doc_id, doc.title, doc.content),
        )
        self._conn.commit()
        self._rebuild_index()
        return doc_id

    def add_text(self, text: str, title: str = "", source: str = "") -> str:
        return self.add_document(Document(title=title, content=text, source=source))

    def hybrid_search(self, query: str, top_k: int = 5) -> list[tuple[Any, float]]:
        """Hybrid search: FTS5 + TF-IDF cosine similarity."""
        # FTS5 score
        fts_results = set()
        try:
            cursor = self._conn.execute(
                "SELECT rowid, rank FROM docs_fts WHERE content MATCH ? ORDER BY rank LIMIT ?",
                (_simple_tokenize(query)[:10], top_k * 2),
            )
            for rowid, _ in cursor.fetchall():
                fts_results.add(rowid - 1)  # rowid is 1-indexed
        except Exception:
            pass

        # TF-IDF cosine similarity
        query_tokens = _simple_tokenize(query)
        query_vec = _compute_tfidf(query_tokens, self._idf)

        scored = []
        for i, chunk in enumerate(self._chunks):
            sim = _cosine_sim(query_vec, chunk.embedding)
            if i in fts_results:
                sim += 0.3  # FTS5 boost
            scored.append((chunk, sim))

        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        results = self.hybrid_search(query, top_k)
        output = []
        for chunk, score in results:
            doc_row = self._conn.execute(
                "SELECT title, source, metadata FROM documents WHERE id = ?", (chunk.doc_id,)
            ).fetchone()
            output.append({
                "text": chunk.text,
                "title": doc_row[0] if doc_row else "",
                "source": doc_row[1] if doc_row else "",
                "score": round(score, 4),
            })
        return output

    def get_relevant_context(self, query: str, top_k: int = 3) -> str:
        results = self.search(query, top_k)
        if not results:
            return ""
        parts = []
        for r in results:
            src = f" [{r['source']}]" if r['source'] else ""
            parts.append(f"> {r['text']}{src}")
        return "\n\n".join(parts)
