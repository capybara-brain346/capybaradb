import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import torch
import numpy as np

from capybaradb.main import CapybaraDB, Index
from capybaradb.storage import Storage


class TestSimpleStorage:
    @pytest.fixture
    def temp_dir(self):
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path)

    @pytest.fixture
    def sample_index(self):
        index = Index()
        index.documents = {"doc1": "Hello world", "doc2": "Test document"}
        index.chunks = {
            "chunk1": {"text": "Hello world", "doc_id": "doc1"},
            "chunk2": {"text": "Test document", "doc_id": "doc2"},
        }
        index.vectors = torch.randn(2, 384)
        index.chunk_ids = ["chunk1", "chunk2"]
        index.total_chunks = 2
        index.total_documents = 2
        index.embedding_dim = 384
        return index

    def test_in_memory_storage(self, sample_index):
        storage = Storage()
        assert storage.in_memory is True
        assert storage.file_path is None

        # Should not raise error
        storage.save(sample_index)
        loaded_index = storage.load()
        assert isinstance(loaded_index, Index)

    def test_disk_storage_save_load(self, temp_dir, sample_index):
        storage_path = temp_dir / "test.npz"
        storage = Storage(storage_path)

        assert storage.in_memory is False
        assert storage.exists() is False

        # Save index
        storage.save(sample_index)
        assert storage.exists() is True

        # Load index
        loaded_index = storage.load()

        # Verify loaded data
        assert loaded_index.documents == sample_index.documents
        assert loaded_index.chunks == sample_index.chunks
        assert torch.equal(loaded_index.vectors, sample_index.vectors)
        assert loaded_index.chunk_ids == sample_index.chunk_ids
        assert loaded_index.total_chunks == sample_index.total_chunks
        assert loaded_index.total_documents == sample_index.total_documents
        assert loaded_index.embedding_dim == sample_index.embedding_dim

    def test_empty_index_save_load(self, temp_dir):
        storage_path = temp_dir / "empty.npz"
        storage = Storage(storage_path)

        empty_index = Index()
        storage.save(empty_index)

        loaded_index = storage.load()
        assert loaded_index.documents == {}
        assert loaded_index.chunks == {}
        assert loaded_index.vectors is None
        assert loaded_index.chunk_ids == []
        assert loaded_index.total_chunks == 0
        assert loaded_index.total_documents == 0
        assert loaded_index.embedding_dim is None

    def test_clear_storage(self, temp_dir, sample_index):
        storage_path = temp_dir / "test.npz"
        storage = Storage(storage_path)

        storage.save(sample_index)
        assert storage.exists() is True

        storage.clear()
        assert storage.exists() is False

    def test_load_nonexistent_file(self, temp_dir):
        storage_path = temp_dir / "nonexistent.npz"
        storage = Storage(storage_path)

        loaded_index = storage.load()
        assert isinstance(loaded_index, Index)
        assert loaded_index.documents == {}


class TestIndexWithStorage:
    @pytest.fixture
    def temp_dir(self):
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path)

    def test_index_with_storage_path(self, temp_dir):
        storage_path = temp_dir / "test.npz"
        index = Index(storage_path)

        assert index.storage.file_path == storage_path
        assert index.storage.in_memory is False
        assert index._is_loaded is False

    def test_index_without_storage_path(self):
        index = Index()

        assert index.storage.in_memory is True
        assert index.storage.file_path is None

    def test_save_load_cycle(self, temp_dir):
        storage_path = temp_dir / "test.npz"
        index = Index(storage_path)

        # Add some data
        index.documents["doc1"] = "Hello world"
        index.chunks["chunk1"] = {"text": "Hello world", "doc_id": "doc1"}
        index.vectors = torch.randn(1, 384)
        index.chunk_ids = ["chunk1"]
        index.total_chunks = 1
        index.total_documents = 1
        index.embedding_dim = 384

        # Save
        index.save()
        assert storage_path.exists()

        # Create new index and load
        new_index = Index(storage_path)
        new_index.load()

        # Verify data
        assert new_index.documents == index.documents
        assert new_index.chunks == index.chunks
        assert torch.equal(new_index.vectors, index.vectors)
        assert new_index.chunk_ids == index.chunk_ids
        assert new_index.total_chunks == index.total_chunks
        assert new_index.total_documents == index.total_documents
        assert new_index.embedding_dim == index.embedding_dim

    def test_clear_index(self, temp_dir):
        storage_path = temp_dir / "test.npz"
        index = Index(storage_path)

        # Add data and save
        index.documents["doc1"] = "Hello world"
        index.save()
        assert storage_path.exists()

        # Clear
        index.clear()
        assert not storage_path.exists()
        assert index.documents == {}
        assert index.chunks == {}
        assert index.vectors is None


class TestCapybaraDBWithStorage:
    @pytest.fixture
    def temp_dir(self):
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path)

    @patch("capybaradb.main.EmbeddingModel")
    @patch("capybaradb.main.setup_logger")
    def test_capybaradb_with_collection(
        self, mock_logger, mock_embedding_model, temp_dir, sample_text
    ):
        mock_logger.return_value = Mock()
        mock_model = Mock()
        mock_embedding_model.return_value = mock_model
        mock_model.embed.return_value = torch.randn(1, 384)

        # Change to temp directory
        import os

        old_cwd = os.getcwd()
        os.chdir(temp_dir)

        try:
            db = CapybaraDB(collection="test_collection")

            # Check storage path
            expected_path = temp_dir / "data" / "test_collection.npz"
            assert db.index.storage.file_path == expected_path
            assert db.index.storage.in_memory is False

            # Add document
            doc_id = db.add_document(sample_text)

            # Check that file was created
            assert expected_path.exists()

            # Verify data was saved
            assert db.index.total_documents == 1
            assert doc_id in db.index.documents

        finally:
            os.chdir(old_cwd)

    @patch("capybaradb.main.EmbeddingModel")
    @patch("capybaradb.main.setup_logger")
    def test_capybaradb_default_collection(
        self, mock_logger, mock_embedding_model, temp_dir
    ):
        mock_logger.return_value = Mock()
        mock_embedding_model.return_value = Mock()

        # Change to temp directory
        import os

        old_cwd = os.getcwd()
        os.chdir(temp_dir)

        try:
            db = CapybaraDB()

            # Check default storage path
            expected_path = temp_dir / "data" / "capybaradb.npz"
            assert db.index.storage.file_path == expected_path

        finally:
            os.chdir(old_cwd)

    @patch("capybaradb.main.EmbeddingModel")
    @patch("capybaradb.main.setup_logger")
    def test_capybaradb_in_memory_mode(self, mock_logger, mock_embedding_model):
        mock_logger.return_value = Mock()
        mock_embedding_model.return_value = Mock()

        # This should still work for backward compatibility
        db = CapybaraDB(collection=None)

        # Should be in-memory mode
        assert db.index.storage.in_memory is True

    @patch("capybaradb.main.EmbeddingModel")
    @patch("capybaradb.main.setup_logger")
    def test_save_load_cycle(
        self, mock_logger, mock_embedding_model, temp_dir, sample_text
    ):
        mock_logger.return_value = Mock()
        mock_model = Mock()
        mock_embedding_model.return_value = mock_model
        mock_model.embed.return_value = torch.randn(1, 384)
        mock_model.search.return_value = (torch.tensor([0]), torch.tensor([0.9]))

        # Change to temp directory
        import os

        old_cwd = os.getcwd()
        os.chdir(temp_dir)

        try:
            # Create database and add document
            db1 = CapybaraDB(collection="test")
            doc_id = db1.add_document(sample_text)

            # Create new database instance (should auto-load)
            db2 = CapybaraDB(collection="test")

            # Verify data was loaded
            assert db2.index.total_documents == 1
            assert doc_id in db2.index.documents

            # Test search
            results = db2.search("test query")
            assert len(results) == 1
            assert results[0]["doc_id"] == doc_id

        finally:
            os.chdir(old_cwd)

    @patch("capybaradb.main.EmbeddingModel")
    @patch("capybaradb.main.setup_logger")
    def test_manual_save_load(
        self, mock_logger, mock_embedding_model, temp_dir, sample_text
    ):
        mock_logger.return_value = Mock()
        mock_model = Mock()
        mock_embedding_model.return_value = mock_model
        mock_model.embed.return_value = torch.randn(1, 384)

        # Change to temp directory
        import os

        old_cwd = os.getcwd()
        os.chdir(temp_dir)

        try:
            # Create database
            db = CapybaraDB(collection="manual_test")

            # Add document
            doc_id = db.add_document(sample_text)

            # Manual save
            db.save()

            # Clear in-memory data
            db.index.documents.clear()
            db.index.chunks.clear()
            db.index.vectors = None
            db.index.total_documents = 0

            # Manual load
            db.load()

            # Verify data was restored
            assert db.index.total_documents == 1
            assert doc_id in db.index.documents

        finally:
            os.chdir(old_cwd)

    @patch("capybaradb.main.EmbeddingModel")
    @patch("capybaradb.main.setup_logger")
    def test_clear_database(
        self, mock_logger, mock_embedding_model, temp_dir, sample_text
    ):
        mock_logger.return_value = Mock()
        mock_model = Mock()
        mock_embedding_model.return_value = mock_model
        mock_model.embed.return_value = torch.randn(1, 384)

        # Change to temp directory
        import os

        old_cwd = os.getcwd()
        os.chdir(temp_dir)

        try:
            # Create database and add document
            db = CapybaraDB(collection="clear_test")
            db.add_document(sample_text)

            storage_path = db.index.storage.file_path
            assert storage_path.exists()

            # Clear database
            db.clear()

            # Verify storage file was deleted
            assert not storage_path.exists()
            assert db.index.total_documents == 0
            assert db.index.documents == {}

        finally:
            os.chdir(old_cwd)
