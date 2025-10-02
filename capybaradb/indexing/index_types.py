from enum import Enum


class IndexType(Enum):
    BM25 = "BM25"
    TFIDF = "TFIDF"
    CONTEXTUAL = "Contextual"
