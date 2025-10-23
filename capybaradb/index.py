from typing import Dict, List, Optional
import torch
from pathlib import Path

from .storage import Storage


class Index:
    def __init__(self, storage_path: Optional[Path] = None) -> None:
        self.documents: Dict[str, str] = {}
        self.chunks: Dict[str, Dict[str, str]] = {}
        self.vectors: Optional[torch.Tensor] = None
        self.chunk_ids: List[str] = []
        self.total_chunks: int = 0
        self.total_documents: int = 0
        self.embedding_dim: Optional[int] = None
        self.storage = Storage(storage_path)
        self._is_loaded = False

    def ensure_vectors_on_device(self, target_device: str) -> None:
        if self.vectors is not None and self.vectors.device.type != target_device:
            self.vectors = self.vectors.to(target_device)

    def save(self) -> None:
        self.storage.save(self)

    def load(self) -> None:
        if not self._is_loaded and self.storage.exists():
            loaded_index = self.storage.load()
            self.documents = loaded_index.documents
            self.chunks = loaded_index.chunks
            self.vectors = loaded_index.vectors
            self.chunk_ids = loaded_index.chunk_ids
            self.total_chunks = loaded_index.total_chunks
            self.total_documents = loaded_index.total_documents
            self.embedding_dim = loaded_index.embedding_dim
            self._is_loaded = True

    def clear(self) -> None:
        self.documents.clear()
        self.chunks.clear()
        self.vectors = None
        self.chunk_ids.clear()
        self.total_chunks = 0
        self.total_documents = 0
        self.embedding_dim = None
        self.storage.clear()
        self._is_loaded = False
