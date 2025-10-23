from pathlib import Path
from typing import Union, List
import pypdf
from docx import Document
from pdf2image import convert_from_path
from PIL import Image
import torch
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_text_from_pdf(
    file_path: Union[str, Path],
    use_ocr: bool = False,
    ocr_model: str = "microsoft/trocr-base-printed",
    ocr_dpi: int = 300,
) -> str:
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    if not file_path.suffix.lower() == ".pdf":
        raise ValueError(f"File is not a PDF: {file_path}")

    try:
        with open(file_path, "rb") as file:
            pdf_reader = pypdf.PdfReader(file)
            text = ""

            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n"

            text = text.strip()

            if not text and use_ocr:
                images = convert_pdf_to_images(file_path, ocr_dpi)
                if images:
                    ocr_processor = OCRProcessor(ocr_model)
                    text = ocr_processor.extract_text_from_images(images)

            return text

    except Exception as e:
        if use_ocr:
            try:
                images = convert_pdf_to_images(file_path, ocr_dpi)
                if images:
                    ocr_processor = OCRProcessor(ocr_model)
                    return ocr_processor.extract_text_from_images(images)
            except Exception as ocr_error:
                raise Exception(
                    f"Both regular and OCR extraction failed: {e}, {ocr_error}"
                )
        else:
            raise Exception(f"Error reading PDF file: {e}")


def extract_text_from_docx(file_path: str) -> str:
    if not Path(file_path).exists():
        raise FileNotFoundError(f"DOCX file not found: {file_path}")

    if not Path(file_path).suffix.lower() == ".docx":
        raise ValueError(f"File is not a DOCX: {file_path}")

    try:
        doc = Document(file_path)
        text = ""

        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"

        return text.strip()

    except Exception as e:
        raise Exception(f"Error processing DOCX file: {e}")


def extract_text_from_txt(file_path: Union[str, Path], encoding: str = "utf-8") -> str:
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"TXT file not found: {file_path}")

    if not file_path.suffix.lower() == ".txt":
        raise ValueError(f"File is not a TXT: {file_path}")

    try:
        with open(file_path, "r", encoding=encoding) as file:
            return file.read().strip()

    except UnicodeDecodeError as e:
        raise UnicodeDecodeError(
            f"Error decoding TXT file with {encoding} encoding: {e}"
        )
    except Exception as e:
        raise Exception(f"Error processing TXT file: {e}")


def extract_text_from_file(
    file_path: Union[str, Path], encoding: str = "utf-8", use_ocr: bool = False
) -> str:
    file_path = Path(file_path)
    file_extension = file_path.suffix.lower()

    if file_extension == ".pdf":
        return extract_text_from_pdf(file_path, use_ocr)
    elif file_extension == ".docx":
        return extract_text_from_docx(file_path)
    elif file_extension == ".txt":
        return extract_text_from_txt(file_path, encoding)
    else:
        raise ValueError(
            f"Unsupported file type: {file_extension}. Supported types: .pdf, .docx, .txt"
        )


def get_supported_extensions() -> list[str]:
    return [".pdf", ".docx", ".txt"]


def is_supported_file(file_path: Union[str, Path]) -> bool:
    file_path = Path(file_path)
    return file_path.suffix.lower() in get_supported_extensions()


class OCRProcessor:
    def __init__(self, model_name: str = "microsoft/trocr-base-printed"):
        self.model_name = model_name
        self.processor = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()

    def _load_model(self):
        try:
            self.processor = TrOCRProcessor.from_pretrained(self.model_name)
            self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name)
            self.model.to(self.device)
        except Exception as e:
            raise

    def extract_text_from_image(self, image: Image.Image) -> str:
        try:
            pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(
                self.device
            )
            generated_ids = self.model.generate(pixel_values)
            generated_text = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]
            return generated_text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from image: {e}")
            return ""

    def extract_text_from_images(self, images: List[Image.Image]) -> str:
        all_text = []
        for i, image in enumerate(images):
            text = self.extract_text_from_image(image)
            if text:
                all_text.append(text)
        return "\n".join(all_text)


def convert_pdf_to_images(
    file_path: Union[str, Path], dpi: int = 300
) -> List[Image.Image]:
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    try:
        images = convert_from_path(file_path, dpi=dpi)
        return images
    except Exception as e:
        raise Exception(f"Failed to convert PDF to images: {e}")
