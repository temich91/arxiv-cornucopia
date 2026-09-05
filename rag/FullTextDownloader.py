from rag_dataclasses import *
import os
import shutil
from utils.paths import *
import arxiv
from urllib.request import urlretrieve


OUTPUT_PATH = ROOT / "temp_pdf_papers"

class FullTextDownloader:
    """Downloads PDFs for the papers selected by the first-stage retriever."""

    def download(self, arxiv_client: arxiv.Client, paper: Paper, output_dir: Path= OUTPUT_PATH) -> Path:
        """Download one paper and return the local PDF path."""
        paper_id = paper.arxiv_id
        search = arxiv.Search(id_list=[paper_id])
        pdf_url = next(arxiv_client.results(search)).pdf_url
        filename, _ = urlretrieve(pdf_url, output_dir / f"{paper_id}.pdf")
        return Path(filename)

    def clean(self, path):
        path.unlink(missing_ok=True)
