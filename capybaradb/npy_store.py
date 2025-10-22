import os
from typing import List, Tuple

import numpy as np


class NpyVectorStore:
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path

    def exists(self) -> bool:
        return os.path.isfile(self.file_path)

    def save(self, vectors: np.ndarray) -> None:
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized = vectors / norms
        np.save(self.file_path, normalized)

    def load(self) -> np.ndarray:
        return np.load(self.file_path)

    def top_k_cosine(self, query: np.ndarray, top_k: int) -> List[Tuple[int, float]]:
        if query.dtype != np.float32:
            query = query.astype(np.float32)
        qn = np.linalg.norm(query)
        if qn == 0:
            return []
        query = query / qn
        mat = self.load()
        scores = mat @ query
        idx = np.argpartition(-scores, min(top_k, scores.shape[0]) - 1)[:top_k]
        subset = [(int(i), float(scores[i])) for i in idx]
        subset.sort(key=lambda x: x[1], reverse=True)
        return subset
