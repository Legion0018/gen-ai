import os
import logging
from typing import Optional

# PDF libraries
import pdfplumber
import pypdf

# DOCX library
from docx import Document

# OCR libraries
import pytesseract
from PIL import Image, ImageFilter, ImageOps
import easyocr

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# Initialize easyocr once (faster than re-creating every time)
EASY_OCR_READER = easyocr.Reader(["en"], gpu=False)


def pdf_to_text(pdf_path: str) -> str:
    """Extract text from PDF (try pdfplumber → fallback to pypdf)."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            texts = [page.extract_text() for page in pdf.pages if page.extract_text()]
            if texts:
                logging.info(f"Extracted text using pdfplumber from {pdf_path}")
                return "\n".join(texts).strip()
            logging.warning(f"pdfplumber found no text in {pdf_path}")
    except Exception as e:
        logging.warning(f"pdfplumber failed on {pdf_path}: {e}")

    try:
        reader = pypdf.PdfReader(pdf_path)
        texts = [page.extract_text() for page in reader.pages if page.extract_text()]
        if texts:
            logging.info(f"Extracted text using pypdf from {pdf_path}")
            return "\n".join(texts).strip()
        logging.warning(f"pypdf found no text in {pdf_path}")
    except Exception as e:
        logging.warning(f"pypdf failed on {pdf_path}: {e}")

    return "Error extracting text from file."


def docx_to_text(docx_path: str) -> str:
    """Extract text from DOCX file."""
    try:
        doc = Document(docx_path)
        paragraphs = [para.text.strip() for para in doc.paragraphs if para.text.strip()]
        if paragraphs:
            logging.info(f"Extracted text using python-docx from {docx_path}")
            return "\n".join(paragraphs).strip()
        logging.warning(f"DOCX file {docx_path} contained no text")
    except Exception as e:
        logging.warning(f"python-docx failed on {docx_path}: {e}")
    return "Error extracting text from file."


def preprocess_image_for_ocr(image_path: str) -> Optional[Image.Image]:
    """Preprocess image for OCR (binarize, denoise, grayscale)."""
    try:
        img = Image.open(image_path)
        img = img.convert("L")  # grayscale
        img = ImageOps.invert(img)  # invert
        img = img.filter(ImageFilter.MedianFilter())  # denoise
        img = img.point(lambda x: 0 if x < 140 else 255, "1")  # binarize
        return img
    except Exception as e:
        logging.warning(f"Image preprocessing failed on {image_path}: {e}")
        return None


def image_to_text(image_path: str, lang: str = "eng") -> str:
    """Extract text from image (try pytesseract → fallback to easyocr)."""
    try:
        img = preprocess_image_for_ocr(image_path) or Image.open(image_path)
        text = pytesseract.image_to_string(img, lang=lang)
        if text.strip():
            logging.info(f"Extracted text using pytesseract from {image_path}")
            return text.strip()
        logging.warning(f"pytesseract extracted no text from {image_path}")
    except Exception as e:
        logging.warning(f"pytesseract failed on {image_path}: {e}")

    try:
        results = EASY_OCR_READER.readtext(image_path, detail=0)
        text = "\n".join(results).strip()
        if text:
            logging.info(f"Extracted text using easyocr from {image_path}")
            return text
        logging.warning(f"easyocr found no text in {image_path}")
    except Exception as e:
        logging.warning(f"easyocr failed on {image_path}: {e}")

    return "Error extracting text from file."


def extract_text(file_path: str) -> str:
    """Dispatch extraction based on file type."""
    ext = os.path.splitext(file_path)[1].lower()
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}

    if ext == ".pdf":
        return pdf_to_text(file_path)
    elif ext == ".docx":
        return docx_to_text(file_path)
    elif ext in image_exts:
        return image_to_text(file_path)
    else:
        logging.warning(f"Unsupported file type: {ext}")
        return "Error extracting text from file."


def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: python ocr.py <file_path>")
        return
    file_path = sys.argv[1]
    text = extract_text(file_path)
    print(f"\nExtracted text from {file_path}:\n")
    print(text)


if __name__ == "__main__":
    main()
