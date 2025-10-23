import pytest
import os
from pathlib import Path
from unittest.mock import patch, Mock
import torch

from capybaradb.main import CapybaraDB
from capybaradb.utils import extract_text_from_file, is_supported_file


class TestIntegrationWithRealFiles:
    @pytest.fixture
    def data_dir(self):
        return Path(__file__).parent / "data"

    @pytest.fixture
    def pdf_file(self, data_dir):
        return data_dir / "Fine-Tuned LLM_SLM Use Cases in TrackML-Backend.pdf"

    @pytest.fixture
    def txt_file(self, data_dir):
        return data_dir / "CNN-Based Classifiers and Fine-Tune.txt"

    @pytest.fixture
    def docx_file(self, data_dir):
        return data_dir / "research-paper-format.docx"

    def test_file_existence(self, pdf_file, txt_file, docx_file):
        assert pdf_file.exists(), f"PDF file not found: {pdf_file}"
        assert txt_file.exists(), f"TXT file not found: {txt_file}"
        assert docx_file.exists(), f"DOCX file not found: {docx_file}"

    def test_file_support_detection(self, pdf_file, txt_file, docx_file):
        assert is_supported_file(pdf_file)
        assert is_supported_file(txt_file)
        assert is_supported_file(docx_file)

    @patch("capybaradb.utils.PyPDF2.PdfReader")
    def test_pdf_text_extraction_integration(self, mock_pdf_reader, pdf_file):
        mock_page = Mock()
        mock_page.extract_text.return_value = "Sample PDF content for testing"
        mock_reader = Mock()
        mock_reader.pages = [mock_page]
        mock_pdf_reader.return_value = mock_reader

        text = extract_text_from_file(pdf_file)
        assert isinstance(text, str)
        assert len(text) > 0

    def test_txt_text_extraction_integration(self, txt_file):
        text = extract_text_from_file(txt_file)
        assert isinstance(text, str)
        assert len(text) > 0
        assert "CNN" in text or "classifier" in text.lower()

    @patch("capybaradb.utils.Document")
    def test_docx_text_extraction_integration(self, mock_document, docx_file):
        mock_doc = Mock()
        mock_paragraph = Mock()
        mock_paragraph.text = "Sample DOCX content for testing"
        mock_doc.paragraphs = [mock_paragraph]
        mock_document.return_value = mock_doc

        text = extract_text_from_file(docx_file)
        assert isinstance(text, str)
        assert len(text) > 0

    @patch("capybaradb.main.EmbeddingModel")
    @patch("capybaradb.main.setup_logger")
    def test_capybaradb_with_real_txt_file(
        self, mock_logger, mock_embedding_model, txt_file
    ):
        mock_logger.return_value = Mock()
        mock_model = Mock()
        mock_embedding_model.return_value = mock_model
        mock_model.embed.return_value = torch.randn(1, 384)
        mock_model.search.return_value = (torch.tensor([0]), torch.tensor([0.9]))

        db = CapybaraDB(chunking=False)

        text = extract_text_from_file(txt_file)
        doc_id = db.add_document(text)

        assert doc_id is not None
        assert db.index.total_documents == 1
        assert len(text) > 0

        results = db.search("machine learning")
        assert len(results) == 1
        assert results[0]["doc_id"] == doc_id

    @patch("capybaradb.main.EmbeddingModel")
    @patch("capybaradb.main.setup_logger")
    def test_capybaradb_with_chunking_real_file(
        self, mock_logger, mock_embedding_model, txt_file
    ):
        mock_logger.return_value = Mock()
        mock_model = Mock()
        mock_embedding_model.return_value = mock_model

        text = extract_text_from_file(txt_file)

        with patch("capybaradb.main.tiktoken") as mock_tiktoken:
            mock_encoder = Mock()
            mock_encoder.encode.return_value = list(range(1000))
            mock_encoder.decode.return_value = "chunk text"
            mock_tiktoken.get_encoding.return_value = mock_encoder

            mock_model.embed.return_value = torch.randn(4, 384)  # 4 chunks

            db = CapybaraDB(chunking=True, chunk_size=256)
            doc_id = db.add_document(text)

            assert doc_id is not None
            assert db.index.total_documents == 1
            assert db.index.total_chunks > 1

    @patch("capybaradb.main.EmbeddingModel")
    @patch("capybaradb.main.setup_logger")
    def test_capybaradb_multiple_documents(
        self, mock_logger, mock_embedding_model, txt_file, pdf_file, docx_file
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

        with patch("capybaradb.utils.extract_text_from_file") as mock_extract:
            mock_extract.side_effect = [
                "Text content from TXT file",
                "Text content from PDF file",
                "Text content from DOCX file",
            ]

            doc_ids = []
            for file_path in [txt_file, pdf_file, docx_file]:
                text = extract_text_from_file(file_path)
                doc_id = db.add_document(text)
                doc_ids.append(doc_id)

            assert db.index.total_documents == 3
            assert len(doc_ids) == 3

            results = db.search("test query", top_k=3)
            assert len(results) == 3

    @patch("capybaradb.main.EmbeddingModel")
    @patch("capybaradb.main.setup_logger")
    def test_capybaradb_precision_modes(
        self, mock_logger, mock_embedding_model, txt_file
    ):
        mock_logger.return_value = Mock()

        text = extract_text_from_file(txt_file)

        for precision in ["float32", "float16", "binary"]:
            mock_model = Mock()
            mock_embedding_model.return_value = mock_model
            mock_model.embed.return_value = torch.randn(1, 384)
            mock_model.search.return_value = (torch.tensor([0]), torch.tensor([0.9]))

            db = CapybaraDB(precision=precision, chunking=False)
            doc_id = db.add_document(text)

            assert doc_id is not None
            assert db.index.total_documents == 1

    @patch("capybaradb.main.EmbeddingModel")
    @patch("capybaradb.main.setup_logger")
    def test_capybaradb_device_modes(self, mock_logger, mock_embedding_model, txt_file):
        mock_logger.return_value = Mock()

        text = extract_text_from_file(txt_file)

        for device in ["cpu", "cuda"]:
            mock_model = Mock()
            mock_embedding_model.return_value = mock_model
            mock_model.embed.return_value = torch.randn(1, 384)
            mock_model.search.return_value = (torch.tensor([0]), torch.tensor([0.9]))

            db = CapybaraDB(device=device, chunking=False)
            doc_id = db.add_document(text)

            assert doc_id is not None
            assert db.index.total_documents == 1

    def test_error_handling_nonexistent_file(self):
        nonexistent_file = Path("nonexistent_file.txt")

        with pytest.raises(FileNotFoundError):
            extract_text_from_file(nonexistent_file)

    def test_error_handling_unsupported_file_type(self, data_dir):
        unsupported_file = data_dir / "test.xyz"
        unsupported_file.touch()

        with pytest.raises(ValueError, match="Unsupported file type"):
            extract_text_from_file(unsupported_file)

    @patch("capybaradb.main.EmbeddingModel")
    @patch("capybaradb.main.setup_logger")
    def test_capybaradb_empty_search(self, mock_logger, mock_embedding_model):
        mock_logger.return_value = Mock()
        mock_embedding_model.return_value = Mock()

        db = CapybaraDB()
        results = db.search("any query")

        assert results == []

    @patch("capybaradb.main.EmbeddingModel")
    @patch("capybaradb.main.setup_logger")
    def test_capybaradb_document_retrieval(
        self, mock_logger, mock_embedding_model, txt_file
    ):
        mock_logger.return_value = Mock()
        mock_model = Mock()
        mock_embedding_model.return_value = mock_model
        mock_model.embed.return_value = torch.randn(1, 384)

        db = CapybaraDB(chunking=False)

        text = extract_text_from_file(txt_file)
        doc_id = db.add_document(text)

        retrieved_text = db.get_document(doc_id)
        assert retrieved_text == text

        nonexistent_doc = db.get_document("nonexistent_id")
        assert nonexistent_doc is None
