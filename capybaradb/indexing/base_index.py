from typing import Any
from abc import ABC, abstractmethod

from .index_types import IndexType


class BaseIndex(ABC):
    def __init__(self, docs: Any) -> None:
        self.docs = docs
        self.index_type: IndexType | None = None

    @abstractmethod
    def search(self, query, top_k=5):
        pass

    @abstractmethod
    def compute_index(self):
        pass
