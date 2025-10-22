import pytest
import uuid
from unittest.mock import Mock, patch, MagicMock
import torch

from capybaradb.main import CapybaraDB, Index


class TestIndex:
    def test_index_init(self):
        index = Index()

        assert index.documents == {}
        assert index.chunks == {}
        assert index.vectors is None
        assert index.chunk_ids == []
        assert index.total_chunks == 0
        assert index.total_documents == 0
        assert index.embedding_dim is None


class TestCapybaraDB:
    @patch("capybaradb.main.EmbeddingModel")
    @patch("capybaradb.main.setup_logger")
    def test_capybaradb_init_default(self, mock_logger, mock_embedding_model):
        mock_logger.return_value = Mock()
        mock_embedding_model.return_value = Mock()

        db = CapybaraDB()

        assert db.chunking is False
        assert db.chunk_size == 512
        assert isinstance(db.index, Index)
        mock_embedding_model.assert_called_once_with(precision="float32", device="cpu")
        mock_logger.assert_called_once()

    @patch("capybaradb.main.EmbeddingModel")
    @patch("capybaradb.main.setup_logger")
    def test_capybaradb_init_custom_params(self, mock_logger, mock_embedding_model):
        mock_logger.return_value = Mock()
        mock_embedding_model.return_value = Mock()

        db = CapybaraDB(
            chunking=True, chunk_size=256, precision="float16", device="cuda"
        )

        assert db.chunking is True
        assert db.chunk_size == 256
        mock_embedding_model.assert_called_once_with(precision="float16", device="cuda")

    @patch("capybaradb.main.EmbeddingModel")
    @patch("capybaradb.main.setup_logger")
    def test_add_document_without_chunking(
        self, mock_logger, mock_embedding_model, sample_text
    ):
        mock_logger.return_value = Mock()
        mock_model = Mock()
        mock_embedding_model.return_value = mock_model
        mock_model.embed.return_value = torch.randn(1, 384)

        db = CapybaraDB(chunking=False)
        doc_id = db.add_document(sample_text)

        assert isinstance(doc_id, str)
        assert doc_id in db.index.documents
        assert db.index.documents[doc_id] == sample_text
        assert db.index.total_documents == 1
        assert len(db.index.chunks) == 1
        assert db.index.total_chunks == 1
        mock_model.embed.assert_called_once()

    @patch("capybaradb.main.EmbeddingModel")
    @patch("capybaradb.main.setup_logger")
    def test_add_document_with_custom_doc_id(
        self, mock_logger, mock_embedding_model, sample_text
    ):
        mock_logger.return_value = Mock()
        mock_model = Mock()
        mock_embedding_model.return_value = mock_model
        mock_model.embed.return_value = torch.randn(1, 384)

        db = CapybaraDB(chunking=False)
        custom_doc_id = "custom_id_123"
        doc_id = db.add_document(sample_text, doc_id=custom_doc_id)

        assert doc_id == custom_doc_id
        assert db.index.documents[custom_doc_id] == sample_text

    @patch("capybaradb.main.tiktoken")
    @patch("capybaradb.main.EmbeddingModel")
    @patch("capybaradb.main.setup_logger")
    def test_add_document_with_chunking(
        self, mock_logger, mock_embedding_model, mock_tiktoken, sample_text
    ):
        mock_logger.return_value = Mock()
        mock_model = Mock()
        mock_embedding_model.return_value = mock_model
        mock_model.embed.return_value = torch.randn(2, 384)

        mock_encoder = Mock()
        mock_encoder.encode.return_value = list(range(1000))  # 1000 tokens
        mock_encoder.decode.return_value = "chunk text"
        mock_tiktoken.get_encoding.return_value = mock_encoder

        db = CapybaraDB(chunking=True, chunk_size=256)
        doc_id = db.add_document(sample_text)

        assert isinstance(doc_id, str)
        assert db.index.total_documents == 1
        assert db.index.total_chunks > 1  # Should be chunked
        mock_tiktoken.get_encoding.assert_called_once_with("cl100k_base")
        mock_model.embed.assert_called_once()

    @patch("capybaradb.main.tiktoken")
    @patch("capybaradb.main.EmbeddingModel")
    @patch("capybaradb.main.setup_logger")
    def test_add_document_chunking_no_tiktoken(
        self, mock_logger, mock_embedding_model, mock_tiktoken, sample_text
    ):
        mock_logger.return_value = Mock()
        mock_embedding_model.return_value = Mock()
        mock_tiktoken = None

        with patch("capybaradb.main.tiktoken", None):
            db = CapybaraDB(chunking=True)

            with pytest.raises(
                RuntimeError, match="tiktoken is required for token-based chunking"
            ):
                db.add_document(sample_text)

    @patch("capybaradb.main.EmbeddingModel")
    @patch("capybaradb.main.setup_logger")
    def test_add_document_initial_vectors(
        self, mock_logger, mock_embedding_model, sample_text
    ):
        mock_logger.return_value = Mock()
        mock_model = Mock()
        mock_embedding = torch.randn(1, 384)
        mock_model.embed.return_value = mock_embedding
        mock_embedding_model.return_value = mock_model

        db = CapybaraDB(chunking=False)
        doc_id = db.add_document(sample_text)

        assert db.index.vectors is not None
        assert torch.equal(db.index.vectors, mock_embedding)
        assert len(db.index.chunk_ids) == 1
        assert db.index.embedding_dim == 384

    @patch("capybaradb.main.EmbeddingModel")
    @patch("capybaradb.main.setup_logger")
    def test_add_document_append_vectors(
        self, mock_logger, mock_embedding_model, sample_text
    ):
        mock_logger.return_value = Mock()
        mock_model = Mock()
        mock_embedding1 = torch.randn(1, 384)
        mock_embedding2 = torch.randn(1, 384)
        mock_model.embed.side_effect = [mock_embedding1, mock_embedding2]
        mock_embedding_model.return_value = mock_model

        db = CapybaraDB(chunking=False)

        doc_id1 = db.add_document(sample_text)
        doc_id2 = db.add_document(sample_text + " more text")

        assert db.index.total_documents == 2
        assert db.index.total_chunks == 2
        assert db.index.vectors.shape[0] == 2
        assert len(db.index.chunk_ids) == 2

    @patch("capybaradb.main.EmbeddingModel")
    @patch("capybaradb.main.setup_logger")
    def test_search_empty_index(self, mock_logger, mock_embedding_model):
        mock_logger.return_value = Mock()
        mock_embedding_model.return_value = Mock()

        db = CapybaraDB()
        results = db.search("test query")

        assert results == []

    @patch("capybaradb.main.EmbeddingModel")
    @patch("capybaradb.main.setup_logger")
    def test_search_with_documents(
        self, mock_logger, mock_embedding_model, sample_text
    ):
        mock_logger.return_value = Mock()
        mock_model = Mock()
        mock_embedding_model.return_value = mock_model
        mock_model.embed.return_value = torch.randn(1, 384)
        mock_model.search.return_value = (torch.tensor([0]), torch.tensor([0.9]))

        db = CapybaraDB(chunking=False)
        doc_id = db.add_document(sample_text)

        results = db.search("test query", top_k=5)

        assert len(results) == 1
        result = results[0]
        assert result["doc_id"] == doc_id
        assert result["text"] == sample_text
        assert result["document"] == sample_text
        assert "chunk_id" in result
        assert "score" in result
        assert abs(result["score"] - 0.9) < 0.11

    @patch("capybaradb.main.EmbeddingModel")
    @patch("capybaradb.main.setup_logger")
    def test_search_multiple_results(
        self, mock_logger, mock_embedding_model, sample_documents
    ):
        mock_logger.return_value = Mock()
        mock_model = Mock()
        mock_embedding_model.return_value = mock_model
        mock_model.embed.return_value = torch.randn(1, 384)
        mock_model.search.return_value = (
            torch.tensor([0, 1, 2]),
            torch.tensor([0.9, 0.8, 0.7]),
        )

        db = CapybaraDB(chunking=False)
        doc_ids = [db.add_document(doc) for doc in sample_documents]

        results = db.search("test query", top_k=3)

        assert len(results) == 3
        for i, result in enumerate(results):
            assert result["doc_id"] == doc_ids[i]
            assert result["text"] == sample_documents[i]
            assert abs(result["score"] - (0.9 - i * 0.1)) < 0.11

    @patch("capybaradb.main.EmbeddingModel")
    @patch("capybaradb.main.setup_logger")
    def test_get_document_existing(
        self, mock_logger, mock_embedding_model, sample_text
    ):
        mock_logger.return_value = Mock()
        mock_model = Mock()
        mock_embedding_model.return_value = mock_model
        mock_model.embed.return_value = torch.randn(1, 384)

        db = CapybaraDB(chunking=False)
        doc_id = db.add_document(sample_text)

        retrieved_text = db.get_document(doc_id)
        assert retrieved_text == sample_text

    @patch("capybaradb.main.EmbeddingModel")
    @patch("capybaradb.main.setup_logger")
    def test_get_document_nonexistent(self, mock_logger, mock_embedding_model):
        mock_logger.return_value = Mock()
        mock_embedding_model.return_value = Mock()

        db = CapybaraDB()
        retrieved_text = db.get_document("nonexistent_id")

        assert retrieved_text is None

    @patch("capybaradb.main.EmbeddingModel")
    @patch("capybaradb.main.setup_logger")
    def test_chunk_metadata(self, mock_logger, mock_embedding_model, sample_text):
        mock_logger.return_value = Mock()
        mock_model = Mock()
        mock_embedding_model.return_value = mock_model
        mock_model.embed.return_value = torch.randn(1, 384)

        db = CapybaraDB(chunking=False)
        doc_id = db.add_document(sample_text)

        chunk_id = list(db.index.chunks.keys())[0]
        chunk_info = db.index.chunks[chunk_id]

        assert chunk_info["text"] == sample_text
        assert chunk_info["doc_id"] == doc_id

    @patch("capybaradb.main.EmbeddingModel")
    @patch("capybaradb.main.setup_logger")
    def test_multiple_documents_tracking(
        self, mock_logger, mock_embedding_model, sample_documents
    ):
        mock_logger.return_value = Mock()
        mock_model = Mock()
        mock_embedding_model.return_value = mock_model
        mock_model.embed.return_value = torch.randn(1, 384)

        db = CapybaraDB(chunking=False)

        for doc in sample_documents:
            db.add_document(doc)

        assert db.index.total_documents == len(sample_documents)
        assert len(db.index.documents) == len(sample_documents)
        assert db.index.total_chunks == len(sample_documents)

    @patch("capybaradb.main.EmbeddingModel")
    @patch("capybaradb.main.setup_logger")
    def test_uuid_generation(self, mock_logger, mock_embedding_model, sample_text):
        mock_logger.return_value = Mock()
        mock_model = Mock()
        mock_embedding_model.return_value = mock_model
        mock_model.embed.return_value = torch.randn(1, 384)

        db = CapybaraDB(chunking=False)

        with patch("uuid.uuid4") as mock_uuid:
            mock_uuid.return_value = Mock()
            mock_uuid.return_value.__str__ = Mock(return_value="test-uuid")

            doc_id = db.add_document(sample_text)
            assert doc_id == "test-uuid"
            assert mock_uuid.call_count == 2

    @patch("capybaradb.main.EmbeddingModel")
    @patch("capybaradb.main.setup_logger")
    def test_search_calls_model_search(
        self, mock_logger, mock_embedding_model, sample_text
    ):
        mock_logger.return_value = Mock()
        mock_model = Mock()
        mock_embedding_model.return_value = mock_model
        mock_model.embed.return_value = torch.randn(1, 384)
        mock_model.search.return_value = (torch.tensor([0]), torch.tensor([0.9]))

        db = CapybaraDB(chunking=False)
        db.add_document(sample_text)

        db.search("test query", top_k=5)

        mock_model.search.assert_called_once_with("test query", db.index.vectors, 5)
