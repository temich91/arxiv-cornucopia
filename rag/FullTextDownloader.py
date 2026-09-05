from rag_dataclasses import *
from pathlib import Path


class FullTextDownloader:
    """Downloads PDFs for the papers selected by the first-stage retriever."""

    def download(self, paper: Paper, output_dir: Path) -> Path:
        """Download one paper and return the local PDF path."""
        raise NotImplementedError
