import io
from pypdf import PdfReader
from PIL import Image
import pytesseract

def extract_pdf(data: bytes):
    reader = PdfReader(io.BytesIO(data))
    text_parts = []
    for page in reader.pages:
        text_parts.append(page.extract_text() or "")
    text = "\n".join(text_parts).strip()
    return {
        "text": text,
        "method": "pdf_text",
        "ocr_required": not bool(text),
    }

def extract_image(data: bytes):
    image = Image.open(io.BytesIO(data))
    text = pytesseract.image_to_string(image).strip()
    return {
        "text": text,
        "method": "ocr",
        "ocr_required": False,
    }
