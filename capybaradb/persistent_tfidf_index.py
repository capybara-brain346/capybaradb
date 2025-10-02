import sqlite3
import math
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Optional
from contextlib import contextmanager

from .indexing.base_index import BaseIndex
from .indexing.index_types import IndexType


class PersistentTFIDFIndex(BaseIndex):
    def __init__(self, db_path: str, docs: Optional[List[str]] = None) -> None:
        super().__init__(docs or [])
        self.db_path = db_path
        self.index_type = IndexType.TFIDF
        self._initialize_db()
        
        if docs:
            self._add_documents(docs)

    def _initialize_db(self) -> None:
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY,
                    text TEXT NOT NULL,
                    doc_len INTEGER NOT NULL,
                    doc_norm REAL NOT NULL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS vocab (
                    term TEXT PRIMARY KEY,
                    df INTEGER NOT NULL,
                    idf REAL NOT NULL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS doc_terms (
                    doc_id INTEGER NOT NULL,
                    term TEXT NOT NULL,
                    tf INTEGER NOT NULL,
                    PRIMARY KEY (doc_id, term),
                    FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS tfidf_config (
                    id INTEGER PRIMARY KEY CHECK (id=1),
                    total_docs INTEGER NOT NULL
                )
                """
            )
            cur.execute(
                """
                INSERT OR IGNORE INTO tfidf_config(id, total_docs)
                VALUES (1, 0)
                """
            )

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _add_documents(self, docs: List[str]) -> None:
        for doc in docs:
            self._add_document(doc)

    def _add_document(self, text: str) -> int:
        tokens = self._tokenize(text)
        doc_id = self._insert_document(text, len(tokens))
        self._update_vocabulary_and_terms(doc_id, tokens)
        return doc_id

    def _tokenize(self, text: str) -> List[str]:
        return text.lower().split()

    def _insert_document(self, text: str, doc_len: int) -> int:
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO documents(text, doc_len, doc_norm) VALUES (?, ?, 0.0)",
                (text, doc_len)
            )
            return cur.lastrowid

    def _update_vocabulary_and_terms(self, doc_id: int, tokens: List[str]) -> None:
        if not tokens:
            return

        term_counts = Counter(tokens)
        unique_terms = list(term_counts.keys())

        with self._connect() as conn:
            cur = conn.cursor()
            
            for term, tf in term_counts.items():
                cur.execute(
                    """
                    INSERT INTO doc_terms(doc_id, term, tf) VALUES (?, ?, ?)
                    ON CONFLICT(doc_id, term) DO UPDATE SET tf=excluded.tf
                    """,
                    (doc_id, term, tf)
                )

            for term in unique_terms:
                cur.execute(
                    """
                    INSERT INTO vocab(term, df, idf) VALUES(?, 1, 0.0)
                    ON CONFLICT(term) DO UPDATE SET df = df + 1
                    """,
                    (term,)
                )

            cur.execute("UPDATE tfidf_config SET total_docs = total_docs + 1 WHERE id = 1")
            
        self._recompute_idf()
        self._recompute_doc_norms()

    def _recompute_idf(self) -> None:
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute("SELECT total_docs FROM tfidf_config WHERE id = 1")
            total_docs = cur.fetchone()[0]
            
            cur.execute("SELECT term, df FROM vocab")
            for row in cur.fetchall():
                term = row["term"]
                df = row["df"]
                idf = 0.0 if df == 0 else math.log(total_docs / df)
                cur.execute("UPDATE vocab SET idf = ? WHERE term = ?", (idf, term))

    def _recompute_doc_norms(self) -> None:
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute("SELECT id FROM documents")
            doc_ids = [row[0] for row in cur.fetchall()]
            
            for doc_id in doc_ids:
                cur.execute(
                    """
                    SELECT dt.term, dt.tf, v.idf
                    FROM doc_terms dt
                    JOIN vocab v ON dt.term = v.term
                    WHERE dt.doc_id = ?
                    """,
                    (doc_id,)
                )
                
                doc_norm = 0.0
                for row in cur.fetchall():
                    tf = row["tf"]
                    idf = row["idf"]
                    w = (1 + math.log(tf)) * idf
                    doc_norm += w * w
                
                doc_norm = math.sqrt(doc_norm)
                cur.execute("UPDATE documents SET doc_norm = ? WHERE id = ?", (doc_norm, doc_id))

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        query_weights = self._compute_query_weights(query_tokens)
        if not query_weights:
            return []

        scores = self._compute_scores(query_weights)
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def _compute_query_weights(self, query_tokens: List[str]) -> Dict[str, float]:
        term_counts = Counter(query_tokens)
        
        with self._connect() as conn:
            cur = conn.cursor()
            placeholders = ",".join(["?"] * len(term_counts))
            terms = list(term_counts.keys())
            
            cur.execute(
                f"SELECT term, idf FROM vocab WHERE term IN ({placeholders})",
                terms
            )
            idf_map = {row["term"]: row["idf"] for row in cur.fetchall()}
        
        query_weights = {}
        for term, tf in term_counts.items():
            if term in idf_map:
                w = (1 + math.log(tf)) * idf_map[term]
                query_weights[term] = w
        
        return query_weights

    def _compute_scores(self, query_weights: Dict[str, float]) -> List[Tuple[int, float]]:
        query_terms = list(query_weights.keys())
        query_norm = math.sqrt(sum(w * w for w in query_weights.values()))
        
        with self._connect() as conn:
            cur = conn.cursor()
            placeholders = ",".join(["?"] * len(query_terms))
            
            cur.execute(
                f"""
                SELECT dt.doc_id, dt.term, dt.tf, v.idf, d.doc_norm
                FROM doc_terms dt
                JOIN vocab v ON dt.term = v.term
                JOIN documents d ON dt.doc_id = d.id
                WHERE dt.term IN ({placeholders})
                """,
                query_terms
            )
            
            doc_scores = defaultdict(float)
            doc_norms = {}
            
            for row in cur.fetchall():
                doc_id = row["doc_id"]
                term = row["term"]
                tf = row["tf"]
                idf = row["idf"]
                doc_norm = row["doc_norm"]
                
                doc_norms[doc_id] = doc_norm
                w_d = (1 + math.log(tf)) * idf
                doc_scores[doc_id] += query_weights[term] * w_d
            
            scores = []
            for doc_id, score in doc_scores.items():
                if doc_norms[doc_id] > 0:
                    cosine_score = score / (query_norm * doc_norms[doc_id])
                    scores.append((doc_id, cosine_score))
            
            return scores

    def add_documents(self, new_docs: List[str]) -> None:
        for doc in new_docs:
            self._add_document(doc)

    def get_document(self, doc_id: int) -> str:
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute("SELECT text FROM documents WHERE id = ?", (doc_id,))
            row = cur.fetchone()
            if row is None:
                raise IndexError(f"Document ID {doc_id} not found")
            return row["text"]

    def get_document_count(self) -> int:
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM documents")
            return cur.fetchone()[0]

    def compute_index(self) -> None:
        pass
