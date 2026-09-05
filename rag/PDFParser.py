from pathlib import Path

class PDFParser:
    """Converts a PDF into plain/structured text."""

    def parse(self, pdf_path: Path) -> str:
        """Return extracted text from a PDF."""

        raise NotImplementedError
