import logging
from typing import Literal, List, Union
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

from .logger import setup_logger


class EmbeddingModel:
    def __init__(
        self,
        precision: Literal["binary", "float16", "float32"] = "float32",
        device: Literal["cpu", "cuda"] = "cpu",
    ) -> None:
        self.logger = setup_logger(self.__class__.__name__, level=logging.DEBUG)
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.device = device
        self.precision = precision

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if precision == "float16":
            self.model = AutoModel.from_pretrained(
                self.model_name, dtype=torch.float16
            ).to(device)
        else:
            self.model = AutoModel.from_pretrained(self.model_name).to(device)

    def embed(self, documents: Union[str, List[str]]) -> torch.Tensor:
        encoded_documents = self.tokenizer(
            documents, padding=True, truncation=True, return_tensors="pt"
        )
        encoded_documents = {k: v.to(self.device) for k, v in encoded_documents.items()}

        with torch.no_grad():
            model_output = self.model(**encoded_documents)

        sentence_embeddings = self._mean_pooling(
            model_output, encoded_documents["attention_mask"]
        )
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        if self.precision == "binary":
            sentence_embeddings = (sentence_embeddings > 0).float()
        elif self.precision == "float16":
            sentence_embeddings = sentence_embeddings.half()

        return sentence_embeddings

    def search(
        self, query: str, embeddings: torch.Tensor, top_k: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query_embedding = self.embed(query)

        if query_embedding.device != embeddings.device:
            query_embedding = query_embedding.to(embeddings.device)

        if self.precision == "binary":
            similarities = torch.matmul(
                embeddings.float(), query_embedding.t().float()
            ) / query_embedding.size(1)
        else:
            similarities = torch.matmul(embeddings, query_embedding.t())

        scores, indices = torch.topk(
            similarities.squeeze(), min(top_k, embeddings.size(0))
        )

        return indices, scores

    def _mean_pooling(self, model_output, attention_mask) -> torch.Tensor:
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )
