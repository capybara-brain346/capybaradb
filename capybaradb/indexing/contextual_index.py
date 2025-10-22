from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import os
from typing import Optional
import numpy as np

from .base_index import BaseIndex
from .index_types import IndexType
from ..npy_store import NpyVectorStore


class ContextualIndex(BaseIndex):
    def __init__(self, docs, persist_dir: Optional[str] = None) -> None:
        super().__init__(docs)
        self.index_type = IndexType.CONTEXTUAL
        self.tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.embeddings = None
        self.persist_dir = persist_dir
        self._npy_store = (
            None
            if persist_dir is None
            else NpyVectorStore(os.path.join(persist_dir, "embeddings.npy"))
        )

    def compute_index(self):
        encoded_input = self._encode()

        with torch.no_grad():
            model_output = self.model(**encoded_input)

        sentence_embeddings = self._mean_pooling(
            model_output, encoded_input["attention_mask"]
        )

        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        self.embeddings = sentence_embeddings
        if self._npy_store is not None:
            arr = self.embeddings.detach().cpu().numpy().astype(np.float32)
            self._npy_store.save(arr)
        return sentence_embeddings

    def search(self, query, top_k=5):
        if self.embeddings is None and self._npy_store is None:
            self.compute_index()

        query_encoded = self.tokenizer(
            [query], padding=True, truncation=True, return_tensors="pt"
        )

        with torch.no_grad():
            query_output = self.model(**query_encoded)

        query_embedding = self._mean_pooling(
            query_output, query_encoded["attention_mask"]
        )
        query_embedding = F.normalize(query_embedding, p=2, dim=1)

        if self.embeddings is not None:
            similarities = F.cosine_similarity(query_embedding, self.embeddings)
            scores = [(i, similarities[i].item()) for i in range(len(self.docs))]
            scores.sort(key=lambda x: x[1], reverse=True)
            return scores[:top_k]

        if self._npy_store is not None and self._npy_store.exists():
            q_vec = query_embedding.squeeze(0).detach().cpu().numpy().astype(np.float32)
            top = self._npy_store.top_k_cosine(q_vec, top_k)
            return top

        self.compute_index()
        similarities = F.cosine_similarity(query_embedding, self.embeddings)
        scores = [(i, similarities[i].item()) for i in range(len(self.docs))]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def _encode(self):
        return self.tokenizer(
            self.docs, padding=True, truncation=True, return_tensors="pt"
        )

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )


if __name__ == "__main__":
    documents = [
        """Machine learning is a subset of artificial intelligence that focuses on 
    building systems that learn from data. Machine learning algorithms use 
    statistical techniques to enable computers to improve their performance 
    on tasks through experience. Deep learning is a specialized form of 
    machine learning that uses neural networks with multiple layers.""",
        """Cooking pasta is simple and delicious. First, boil water in a large pot 
    and add salt. Then add the pasta and cook for eight to ten minutes. 
    Drain the pasta and add your favorite sauce. Italian cooking emphasizes 
    fresh ingredients and simple preparation techniques. Pasta dishes are 
    beloved around the world.""",
        """Space exploration has advanced significantly in recent decades. NASA and 
    private companies are developing new technologies for Mars missions. 
    Rockets are becoming reusable, reducing the cost of space travel. 
    Scientists study planets, stars, and galaxies to understand our universe. 
    The International Space Station orbits Earth as a research laboratory.""",
        """Climate change poses significant challenges to our planet. Rising 
    temperatures affect weather patterns and sea levels. Greenhouse gases 
    from human activities trap heat in the atmosphere. Scientists emphasize 
    the need for renewable energy and sustainable practices. Global 
    cooperation is essential to address climate change effectively.""",
        """Ancient civilizations built remarkable structures that still stand today. 
    The pyramids of Egypt showcase advanced engineering knowledge. Roman 
    aqueducts supplied water to cities across their empire. Ancient Greeks 
    made significant contributions to philosophy, mathematics, and democracy. 
    Archaeologists continue discovering artifacts that reveal secrets of 
    ancient cultures.""",
    ]
    contextual_embeddings = ContextualIndex(documents)
    contextual_embeddings.compute_index()
