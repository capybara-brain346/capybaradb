import numpy as np
import torch
from pathlib import Path
from typing import Optional
import logging

from .logger import setup_logger
from .base import BaseIndex


class Storage:
    def __init__(self, file_path: Optional[Path] = None):
        self.file_path = file_path
        self.in_memory = file_path is None
        self.logger = setup_logger(self.__class__.__name__, level=logging.DEBUG)

    def save(self, index) -> None:
        if self.in_memory:
            self.logger.debug("In-memory mode: skipping save")
            return

        try:
            data = {
                "vectors": index.vectors.cpu().numpy()
                if index.vectors is not None
                else np.array([]),
                "chunk_ids": np.array(index.chunk_ids),
                "chunk_texts": np.array(
                    [index.chunks[cid]["text"] for cid in index.chunk_ids]
                )
                if index.chunk_ids
                else np.array([]),
                "chunk_doc_ids": np.array(
                    [index.chunks[cid]["doc_id"] for cid in index.chunk_ids]
                )
                if index.chunk_ids
                else np.array([]),
                "doc_ids": np.array(list(index.documents.keys())),
                "doc_texts": np.array(list(index.documents.values())),
                "total_chunks": index.total_chunks,
                "total_documents": index.total_documents,
                "embedding_dim": index.embedding_dim or 0,
            }

            self.file_path.parent.mkdir(parents=True, exist_ok=True)

            np.savez_compressed(self.file_path, **data)
            self.logger.info(f"Index saved to {self.file_path}")

        except Exception as e:
            self.logger.error(f"Failed to save index: {e}")
            raise

    def load(self) -> BaseIndex:
        if self.in_memory or not self.exists():
            self.logger.debug(
                "In-memory mode or file doesn't exist: returning empty index"
            )
            return BaseIndex()

        try:
            data = np.load(self.file_path)

            index = BaseIndex()

            if len(data["vectors"]) > 0:
                index.vectors = torch.from_numpy(data["vectors"])
            else:
                index.vectors = None

            index.chunk_ids = data["chunk_ids"].tolist()

            for i, chunk_id in enumerate(index.chunk_ids):
                index.chunks[chunk_id] = {
                    "text": data["chunk_texts"][i],
                    "doc_id": data["chunk_doc_ids"][i],
                }

            for i, doc_id in enumerate(data["doc_ids"]):
                index.documents[doc_id] = data["doc_texts"][i]

            index.total_chunks = int(data["total_chunks"])
            index.total_documents = int(data["total_documents"])
            index.embedding_dim = (
                int(data["embedding_dim"]) if data["embedding_dim"] > 0 else None
            )

            self.logger.info(f"Index loaded from {self.file_path}")
            return index

        except Exception as e:
            self.logger.error(f"Failed to load index: {e}")
            raise

    def exists(self) -> bool:
        return self.file_path is not None and self.file_path.exists()

    def clear(self) -> None:
        if self.file_path and self.file_path.exists():
            self.file_path.unlink()
            self.logger.info(f"Storage file cleared: {self.file_path}")
