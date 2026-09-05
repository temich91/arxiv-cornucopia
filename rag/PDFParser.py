from pathlib import Path
import pymupdf
import re

class PDFParser:
    """Converts a PDF into plain/structured text."""
    def parse(self, pdf_path: Path) -> str:
        """Return extracted text from a PDF without title and references pages."""
        try:
            doc = pymupdf.open(pdf_path)
            full_text_list = []

            for page in doc:
                text = page.get_text("text", sort=True)
                full_text_list.append(text)

            full_text = "\n".join(full_text_list)

            start_match = re.search(r'\b(Abstract|Introduction)\b', full_text, re.IGNORECASE)
            if start_match:
                full_text = full_text[start_match.start():]

            end_match = re.search(r'\n\s*\d*\.?\s*(References|Bibliography)\b', full_text, re.IGNORECASE)
            if end_match:
                full_text = full_text[:end_match.start()]

            return full_text.strip()

        except Exception as e:
            print(f"{pdf_path}: {e}")
            return None
