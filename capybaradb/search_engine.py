from .vectors import bm25_index, contextual_index, tfidf_index
from .vectors.base_index import BaseIndex


class SearchEngine:
    def __init__(
        self,
        index: tfidf_index.TFIDFIndex
        | bm25_index.BM25Index
        | contextual_index.ContextualIndex,
    ) -> None:
        self.index = index
        if not isinstance(self.index, BaseIndex):
            raise TypeError("Index must be an instance of BaseIndex")

    def search(self, query: str, top_k: int = 5):
        return self.index.search(query, top_k)

    def add_documents(self, new_docs: list[str]):
        self.index.docs.extend(new_docs)
        self.index.compute_index()

    def get_document(self, doc_id: int):
        if 0 <= doc_id < len(self.index.docs):
            return self.index.docs[doc_id]
        raise IndexError(f"Document ID {doc_id} out of range")

    def get_index_type(self):
        return self.index.index_type

    def get_document_count(self):
        return len(self.index.docs)
