import logging
from .logger import setup_logger


from typing import Literal
from .model import EmbeddingModel


class CapybaraDB:
    def __init__(
        self,
        chunking: bool,
        chunk_size: int = 512,
        precision: Literal["binary", "float16", "float32"] = "float32",
        device: Literal["cpu", "cuda"] = "cpu",
    ) -> None:
        self.logger = setup_logger(self.__class__.__name__, level=logging.DEBUG)
        self.chunking = chunking
        self.chunk_size = chunk_size
        self.model = EmbeddingModel(precision=precision, device=device)
