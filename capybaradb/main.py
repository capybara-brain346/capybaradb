import logging
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Union

import tiktoken
import torch
from typing_extensions import Literal

from .base import BaseIndex
from .logger import setup_logger
from .model import EmbeddingModel
from .storage import Storage


class Index(BaseIndex):
    def __init__(self, storage_path: Optional[Path] = None) -> None:
        super().__init__()
        self.storage = Storage(storage_path)
        self._is_loaded = False

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


class CapybaraDB:
    def __init__(
        self,
        collection: Optional[str] = None,
        chunking: bool = False,
        chunk_size: int = 512,
        precision: Literal["binary", "float16", "float32"] = "float32",
        device: Literal["cpu", "cuda"] = "cpu",
    ) -> None:
        self.logger = setup_logger(self.__class__.__name__, level=logging.DEBUG)
        self.chunking = chunking
        self.chunk_size = chunk_size

        storage_path = None
        if collection:
            storage_path = Path("data") / f"{collection}.npz"
            self.logger.info(f"Using collection: {collection} -> {storage_path}")
        else:
            storage_path = Path("data") / "capybaradb.npz"
            self.logger.info(f"Using default collection: capybaradb -> {storage_path}")

        self.index = Index(storage_path)
        self.model = EmbeddingModel(precision=precision, device=device)

        if storage_path and self.index.storage.exists():
            self.logger.info("Auto-loading existing collection")
            self.index.load()

    def save(self) -> None:
        self.index.save()

    def load(self) -> None:
        self.index.load()

    def clear(self) -> None:
        self.index.clear()

    def add_document(self, text: str, doc_id: Optional[str] = None) -> str:
        if doc_id is None:
            doc_id = str(uuid.uuid4())

        self.index.documents[doc_id] = text
        self.index.total_documents += 1

        if self.chunking:
            if tiktoken is None:
                raise RuntimeError(
                    "tiktoken is required for token-based chunking. Install with `pip install tiktoken`"
                )
            enc = tiktoken.get_encoding("cl100k_base")
            token_ids = enc.encode(text)
            chunks = []
            for i in range(0, len(token_ids), self.chunk_size):
                tok_chunk = token_ids[i : i + self.chunk_size]
                chunk_text = enc.decode(tok_chunk)
                chunks.append(chunk_text)
        else:
            chunks = [text]

        chunk_ids = []
        for chunk in chunks:
            chunk_id = str(uuid.uuid4())
            self.index.chunks[chunk_id] = {"text": chunk, "doc_id": doc_id}
            chunk_ids.append(chunk_id)
            self.index.total_chunks += 1

        chunk_texts = [self.index.chunks[cid]["text"] for cid in chunk_ids]
        chunk_embeddings = self.model.embed(chunk_texts)

        target_device = self.model.device
        self.index.ensure_vectors_on_device(target_device)

        if self.index.vectors is None:
            self.index.vectors = chunk_embeddings
            self.index.chunk_ids = chunk_ids
            self.index.embedding_dim = chunk_embeddings.size(1)
        else:
            if self.index.vectors.device != chunk_embeddings.device:
                chunk_embeddings = chunk_embeddings.to(self.index.vectors.device)

            self.index.vectors = torch.cat(
                [self.index.vectors, chunk_embeddings], dim=0
            )
            self.index.chunk_ids.extend(chunk_ids)

        if not self.index.storage.in_memory:
            self.index.save()

        return doc_id

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Union[str, float]]]:
        if self.index.vectors is None:
            return []

        target_device = self.model.device
        self.index.ensure_vectors_on_device(target_device)

        indices, scores = self.model.search(query, self.index.vectors, top_k)

        results = []
        for idx, score in zip(indices.tolist(), scores.tolist()):
            chunk_id = self.index.chunk_ids[idx]
            chunk_info = self.index.chunks[chunk_id]
            doc_id = chunk_info["doc_id"]

            results.append(
                {
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "text": chunk_info["text"],
                    "score": score,
                    "document": self.index.documents[doc_id],
                }
            )

        return results

    def get_document(self, doc_id: str) -> Optional[str]:
        return self.index.documents.get(doc_id)
