import logging
from .logger import setup_logger
from typing import Dict, List, Union, Optional
import torch
import uuid
from .model import EmbeddingModel
from typing_extensions import Literal

try:
    import tiktoken
except Exception:
    tiktoken = None


class Index:
    def __init__(self) -> None:
        self.documents: Dict[str, str] = {}
        self.chunks: Dict[str, Dict[str, str]] = {}
        self.vectors: Optional[torch.Tensor] = None
        self.chunk_ids: List[str] = []
        self.total_chunks: int = 0
        self.total_documents: int = 0
        self.embedding_dim: Optional[int] = None


class CapybaraDB:
    def __init__(
        self,
        chunking: bool = False,
        chunk_size: int = 512,
        precision: Literal["binary", "float16", "float32"] = "float32",
        device: Literal["cpu", "cuda"] = "cpu",
    ) -> None:
        self.logger = setup_logger(self.__class__.__name__, level=logging.DEBUG)
        self.chunking = chunking
        self.chunk_size = chunk_size
        self.index = Index()
        self.model = EmbeddingModel(precision=precision, device=device)

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

        if self.index.vectors is None:
            self.index.vectors = chunk_embeddings
            self.index.chunk_ids = chunk_ids
            self.index.embedding_dim = chunk_embeddings.size(1)
        else:
            self.index.vectors = torch.cat(
                [self.index.vectors, chunk_embeddings], dim=0
            )
            self.index.chunk_ids.extend(chunk_ids)

        return doc_id

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Union[str, float]]]:
        if self.index.vectors is None:
            return []

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
