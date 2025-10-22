import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import PyPDF2
from docx import Document

from capybaradb.utils import (
    extract_text_from_pdf,
    extract_text_from_docx,
    extract_text_from_txt,
    extract_text_from_file,
    get_supported_extensions,
    is_supported_file,
    OCRProcessor,
    convert_pdf_to_images,
)


class TestTextExtraction:
    def test_extract_text_from_txt_success(self, temp_dir, sample_text):
        txt_file = temp_dir / "test.txt"
        txt_file.write_text(sample_text)

        result = extract_text_from_txt(txt_file)
        assert result == sample_text

    def test_extract_text_from_txt_file_not_found(self, temp_dir):
        non_existent_file = temp_dir / "nonexistent.txt"

        with pytest.raises(FileNotFoundError):
            extract_text_from_txt(non_existent_file)

    def test_extract_text_from_txt_wrong_extension(self, temp_dir, sample_text):
        pdf_file = temp_dir / "test.pdf"
        pdf_file.write_text(sample_text)

        with pytest.raises(ValueError, match="File is not a TXT"):
            extract_text_from_txt(pdf_file)

    def test_extract_text_from_txt_encoding_error(self, temp_dir):
        txt_file = temp_dir / "test.txt"
        txt_file.write_bytes(b"\xff\xfe\x00\x00")

        with pytest.raises(Exception):
            extract_text_from_txt(txt_file)

    def test_extract_text_from_txt_custom_encoding(self, temp_dir):
        txt_file = temp_dir / "test.txt"
        content = "Test content with special chars: àáâãäå"
        txt_file.write_text(content, encoding="utf-8")

        result = extract_text_from_txt(txt_file, encoding="utf-8")
        assert result == content

    @patch("capybaradb.utils.Document")
    def test_extract_text_from_docx_success(self, mock_document, temp_dir, sample_text):
        docx_file = temp_dir / "test.docx"
        docx_file.touch()

        mock_doc = Mock()
        mock_paragraph = Mock()
        mock_paragraph.text = sample_text
        mock_doc.paragraphs = [mock_paragraph]
        mock_document.return_value = mock_doc

        result = extract_text_from_docx(docx_file)
        assert result == sample_text

    def test_extract_text_from_docx_file_not_found(self, temp_dir):
        non_existent_file = temp_dir / "nonexistent.docx"

        with pytest.raises(FileNotFoundError):
            extract_text_from_docx(non_existent_file)

    def test_extract_text_from_docx_wrong_extension(self, temp_dir):
        txt_file = temp_dir / "test.txt"
        txt_file.touch()

        with pytest.raises(ValueError, match="File is not a DOCX"):
            extract_text_from_docx(txt_file)

    @patch("capybaradb.utils.Document")
    def test_extract_text_from_docx_exception(self, mock_document, temp_dir):
        docx_file = temp_dir / "test.docx"
        docx_file.touch()

        mock_document.side_effect = Exception("Test error")

        with pytest.raises(Exception, match="Error processing DOCX file"):
            extract_text_from_docx(docx_file)

    @patch("capybaradb.utils.PyPDF2.PdfReader")
    def test_extract_text_from_pdf_success(
        self, mock_pdf_reader, temp_dir, sample_text
    ):
        pdf_file = temp_dir / "test.pdf"
        pdf_file.touch()

        mock_page = Mock()
        mock_page.extract_text.return_value = sample_text
        mock_reader = Mock()
        mock_reader.pages = [mock_page]
        mock_pdf_reader.return_value = mock_reader

        result = extract_text_from_pdf(pdf_file)
        assert result == sample_text

    def test_extract_text_from_pdf_file_not_found(self, temp_dir):
        non_existent_file = temp_dir / "nonexistent.pdf"

        with pytest.raises(FileNotFoundError):
            extract_text_from_pdf(non_existent_file)

    def test_extract_text_from_pdf_wrong_extension(self, temp_dir):
        txt_file = temp_dir / "test.txt"
        txt_file.touch()

        with pytest.raises(ValueError, match="File is not a PDF"):
            extract_text_from_pdf(txt_file)

    @patch("capybaradb.utils.PyPDF2.PdfReader")
    def test_extract_text_from_pdf_read_error_without_ocr(
        self, mock_pdf_reader, temp_dir
    ):
        pdf_file = temp_dir / "test.pdf"
        pdf_file.touch()

        mock_pdf_reader.side_effect = PyPDF2.errors.PdfReadError("Test error")

        with pytest.raises(PyPDF2.errors.PdfReadError):
            extract_text_from_pdf(pdf_file, use_ocr=False)

    @patch("capybaradb.utils.convert_pdf_to_images")
    @patch("capybaradb.utils.OCRProcessor")
    @patch("capybaradb.utils.PyPDF2.PdfReader")
    def test_extract_text_from_pdf_with_ocr_fallback(
        self, mock_pdf_reader, mock_ocr_class, mock_convert, temp_dir
    ):
        pdf_file = temp_dir / "test.pdf"
        pdf_file.touch()

        mock_pdf_reader.side_effect = PyPDF2.errors.PdfReadError("Test error")

        mock_image = Mock(spec=Image.Image)
        mock_convert.return_value = [mock_image]

        mock_ocr = Mock()
        mock_ocr.extract_text_from_images.return_value = "OCR extracted text"
        mock_ocr_class.return_value = mock_ocr

        result = extract_text_from_pdf(pdf_file, use_ocr=True)
        assert result == "OCR extracted text"

    def test_extract_text_from_file_pdf(self, temp_dir, sample_text):
        pdf_file = temp_dir / "test.pdf"
        pdf_file.touch()

        with patch(
            "capybaradb.utils.extract_text_from_pdf", return_value=sample_text
        ) as mock_extract:
            result = extract_text_from_file(pdf_file)
            mock_extract.assert_called_once_with(pdf_file, False)
            assert result == sample_text

    def test_extract_text_from_file_docx(self, temp_dir, sample_text):
        docx_file = temp_dir / "test.docx"
        docx_file.touch()

        with patch(
            "capybaradb.utils.extract_text_from_docx", return_value=sample_text
        ) as mock_extract:
            result = extract_text_from_file(docx_file)
            mock_extract.assert_called_once_with(docx_file)
            assert result == sample_text

    def test_extract_text_from_file_txt(self, temp_dir, sample_text):
        txt_file = temp_dir / "test.txt"
        txt_file.touch()

        with patch(
            "capybaradb.utils.extract_text_from_txt", return_value=sample_text
        ) as mock_extract:
            result = extract_text_from_file(txt_file)
            mock_extract.assert_called_once_with(txt_file, "utf-8")
            assert result == sample_text

    def test_extract_text_from_file_unsupported_type(self, temp_dir):
        unsupported_file = temp_dir / "test.xyz"
        unsupported_file.touch()

        with pytest.raises(ValueError, match="Unsupported file type"):
            extract_text_from_file(unsupported_file)

    def test_get_supported_extensions(self):
        extensions = get_supported_extensions()
        assert extensions == [".pdf", ".docx", ".txt"]

    def test_is_supported_file(self, temp_dir):
        supported_files = [
            temp_dir / "test.pdf",
            temp_dir / "test.docx",
            temp_dir / "test.txt",
        ]

        unsupported_files = [
            temp_dir / "test.xyz",
            temp_dir / "test.doc",
            temp_dir / "test.rtf",
        ]

        for file_path in supported_files:
            assert is_supported_file(file_path)

        for file_path in unsupported_files:
            assert not is_supported_file(file_path)


class TestOCRProcessor:
    @patch("capybaradb.utils.torch.cuda.is_available", return_value=False)
    @patch("capybaradb.utils.TrOCRProcessor")
    @patch("capybaradb.utils.VisionEncoderDecoderModel")
    def test_ocr_processor_init_cpu(self, mock_vision_model, mock_processor, mock_cuda):
        processor = OCRProcessor()

        assert processor.device == "cpu"
        assert processor.model_name == "microsoft/trocr-base-printed"
        mock_processor.from_pretrained.assert_called_once_with(
            "microsoft/trocr-base-printed"
        )
        mock_vision_model.from_pretrained.assert_called_once_with(
            "microsoft/trocr-base-printed"
        )

    @patch("capybaradb.utils.torch.cuda.is_available", return_value=True)
    @patch("capybaradb.utils.TrOCRProcessor")
    @patch("capybaradb.utils.VisionEncoderDecoderModel")
    def test_ocr_processor_init_cuda(
        self, mock_vision_model, mock_processor, mock_cuda
    ):
        processor = OCRProcessor()

        assert processor.device == "cuda"
        mock_vision_model.from_pretrained.return_value.to.assert_called_once_with(
            "cuda"
        )

    @patch("capybaradb.utils.torch.cuda.is_available", return_value=False)
    @patch("capybaradb.utils.TrOCRProcessor")
    @patch("capybaradb.utils.VisionEncoderDecoderModel")
    def test_ocr_processor_init_custom_model(
        self, mock_vision_model, mock_processor, mock_cuda
    ):
        custom_model = "custom/model"
        processor = OCRProcessor(custom_model)

        assert processor.model_name == custom_model
        mock_processor.from_pretrained.assert_called_once_with(custom_model)
        mock_vision_model.from_pretrained.assert_called_once_with(custom_model)

    @patch("capybaradb.utils.torch.cuda.is_available", return_value=False)
    @patch("capybaradb.utils.TrOCRProcessor")
    @patch("capybaradb.utils.VisionEncoderDecoderModel")
    def test_extract_text_from_image_success(
        self, mock_vision_model, mock_processor, mock_cuda
    ):
        mock_processor_instance = MagicMock()
        mock_processor.from_pretrained.return_value = mock_processor_instance
        mock_processor_instance.return_value = MagicMock()
        mock_processor_instance.batch_decode.return_value = ["extracted text"]

        mock_vision_model_instance = MagicMock()
        mock_vision_model.from_pretrained.return_value = mock_vision_model_instance
        mock_vision_model_instance.generate.return_value = MagicMock()

        processor = OCRProcessor()

        mock_image = Mock(spec=Image.Image)

        result = processor.extract_text_from_image(mock_image)
        assert result == "extracted text"

    @patch("capybaradb.utils.torch.cuda.is_available", return_value=False)
    @patch("capybaradb.utils.TrOCRProcessor")
    @patch("capybaradb.utils.VisionEncoderDecoderModel")
    def test_extract_text_from_image_error(
        self, mock_vision_model, mock_processor, mock_cuda
    ):
        mock_processor_instance = MagicMock()
        mock_processor.from_pretrained.return_value = mock_processor_instance
        mock_processor_instance.return_value.side_effect = Exception("Test error")

        mock_vision_model_instance = MagicMock()
        mock_vision_model.from_pretrained.return_value = mock_vision_model_instance

        processor = OCRProcessor()

        mock_image = Mock(spec=Image.Image)

        result = processor.extract_text_from_image(mock_image)
        assert result == ""

    @patch("capybaradb.utils.torch.cuda.is_available", return_value=False)
    @patch("capybaradb.utils.TrOCRProcessor")
    @patch("capybaradb.utils.VisionEncoderDecoderModel")
    def test_extract_text_from_images(
        self, mock_vision_model, mock_processor, mock_cuda
    ):
        processor = OCRProcessor()

        mock_image1 = Mock(spec=Image.Image)
        mock_image2 = Mock(spec=Image.Image)
        images = [mock_image1, mock_image2]

        processor.extract_text_from_image = Mock(side_effect=["text1", "text2"])

        result = processor.extract_text_from_images(images)
        assert result == "text1\ntext2"
        assert processor.extract_text_from_image.call_count == 2

    @patch("capybaradb.utils.torch.cuda.is_available", return_value=False)
    @patch("capybaradb.utils.TrOCRProcessor")
    @patch("capybaradb.utils.VisionEncoderDecoderModel")
    def test_extract_text_from_images_empty_text(
        self, mock_vision_model, mock_processor, mock_cuda
    ):
        processor = OCRProcessor()

        mock_image1 = Mock(spec=Image.Image)
        mock_image2 = Mock(spec=Image.Image)
        images = [mock_image1, mock_image2]

        processor.extract_text_from_image = Mock(side_effect=["text1", ""])

        result = processor.extract_text_from_images(images)
        assert result == "text1"


class TestPDFToImages:
    @patch("capybaradb.utils.convert_from_path")
    def test_convert_pdf_to_images_success(self, mock_convert, temp_dir):
        pdf_file = temp_dir / "test.pdf"
        pdf_file.touch()

        mock_images = [Mock(spec=Image.Image), Mock(spec=Image.Image)]
        mock_convert.return_value = mock_images

        result = convert_pdf_to_images(pdf_file)
        assert result == mock_images
        mock_convert.assert_called_once_with(pdf_file, dpi=300)

    def test_convert_pdf_to_images_file_not_found(self, temp_dir):
        non_existent_file = temp_dir / "nonexistent.pdf"

        with pytest.raises(FileNotFoundError):
            convert_pdf_to_images(non_existent_file)

    @patch("capybaradb.utils.convert_from_path")
    def test_convert_pdf_to_images_custom_dpi(self, mock_convert, temp_dir):
        pdf_file = temp_dir / "test.pdf"
        pdf_file.touch()

        mock_images = [Mock(spec=Image.Image)]
        mock_convert.return_value = mock_images

        result = convert_pdf_to_images(pdf_file, dpi=600)
        mock_convert.assert_called_once_with(pdf_file, dpi=600)

    @patch("capybaradb.utils.convert_from_path")
    def test_convert_pdf_to_images_exception(self, mock_convert, temp_dir):
        pdf_file = temp_dir / "test.pdf"
        pdf_file.touch()

        mock_convert.side_effect = Exception("Conversion failed")

        with pytest.raises(Exception, match="Failed to convert PDF to images"):
            convert_pdf_to_images(pdf_file)
