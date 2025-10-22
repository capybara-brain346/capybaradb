import sqlite3
from contextlib import contextmanager
from typing import Generator, List, Optional, Tuple
import numpy as np


class EmbeddingsDB:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def initialize(self) -> None:
        with self.connect() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    doc_id INTEGER PRIMARY KEY,
                    dim INTEGER NOT NULL,
                    norm REAL NOT NULL,
                    vector BLOB NOT NULL
                )
                """
            )

    def upsert(self, doc_id: int, vector: np.ndarray) -> None:
        if vector.dtype != np.float32:
            vector = vector.astype(np.float32)
        dim = int(vector.shape[-1])
        norm = float(np.linalg.norm(vector))
        blob = vector.tobytes(order="C")
        with self.connect() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO embeddings(doc_id, dim, norm, vector) VALUES (?, ?, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET dim=excluded.dim, norm=excluded.norm, vector=excluded.vector
                """,
                (doc_id, dim, norm, sqlite3.Binary(blob)),
            )

    def get(self, doc_id: int) -> Optional[np.ndarray]:
        with self.connect() as conn:
            cur = conn.cursor()
            cur.execute("SELECT dim, vector FROM embeddings WHERE doc_id = ?", (doc_id,))
            row = cur.fetchone()
            if row is None:
                return None
            dim = int(row["dim"])
            buf = row["vector"]
            arr = np.frombuffer(buf, dtype=np.float32)
            return arr.reshape((dim,))

    def iterate(self, batch_size: int = 1024) -> Generator[List[Tuple[int, np.ndarray]], None, None]:
        offset = 0
        while True:
            with self.connect() as conn:
                cur = conn.cursor()
                cur.execute(
                    "SELECT doc_id, dim, vector FROM embeddings ORDER BY doc_id LIMIT ? OFFSET ?",
                    (batch_size, offset),
                )
                rows = cur.fetchall()
            if not rows:
                break
            batch: List[Tuple[int, np.ndarray]] = []
            for r in rows:
                doc_id = int(r["doc_id"])
                dim = int(r["dim"])
                buf = r["vector"]
                vec = np.frombuffer(buf, dtype=np.float32).reshape((dim,))
                batch.append((doc_id, vec))
            yield batch
            offset += len(rows)

    def delete(self, doc_id: int) -> None:
        with self.connect() as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM embeddings WHERE doc_id = ?", (doc_id,))

    def count(self) -> int:
        with self.connect() as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(1) FROM embeddings")
            return int(cur.fetchone()[0])


