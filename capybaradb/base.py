from typing import Dict, List, Optional
import torch
from pathlib import Path


class BaseIndex:
    def __init__(self) -> None:
        self.documents: Dict[str, str] = {}
        self.chunks: Dict[str, Dict[str, str]] = {}
        self.vectors: Optional[torch.Tensor] = None
        self.chunk_ids: List[str] = []
        self.total_chunks: int = 0
        self.total_documents: int = 0
        self.embedding_dim: Optional[int] = None
        self._is_loaded = False

    def ensure_vectors_on_device(self, target_device: str) -> None:
        if self.vectors is not None and self.vectors.device.type != target_device:
            self.vectors = self.vectors.to(target_device)
