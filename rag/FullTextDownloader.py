from rag_dataclasses import *
import shutil
from pathlib import Path
from urllib.request import urlretrieve


class FullTextDownloader:
    """Downloads PDFs for the papers selected by the first-stage retriever."""

    def download(self, paper: Paper, output_dir: Path) -> Path:
        """Download one paper and return the local PDF path."""
        paper_id = paper.arxiv_id
        url = paper.pdf_url
        filename, _ = urlretrieve(url, output_dir / f"{paper_id}.pdf")
        return Path(filename)

    def clean(self, path):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            for item in path.iterdir():
                if item.is_file() or item.is_symlink():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
