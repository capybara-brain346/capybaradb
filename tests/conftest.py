import pytest
import tempfile
from pathlib import Path


@pytest.fixture
def sample_text():
    return "This is a sample text for testing purposes."


@pytest.fixture
def sample_documents():
    return [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "Natural language processing deals with text and speech.",
    ]


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)
