from ArxivRetriever import ArxivRetriever
from FullTextDownloader import FullTextDownloader
from PDFParser import PDFParser
from TextChunker import TextChunker
from TextsReranker import FullTextReranker
from pathlib import Path
from rag_dataclasses import *


class ResearchPipeline:
    """Coordinates the complete two-stage retrieval pipeline."""

    def __init__(
        self,
        retriever: ArxivRetriever,
        downloader: FullTextDownloader,
        parser: PDFParser,
        chunker: TextChunker,
        reranker: FullTextReranker,
    ):
        self.retriever = retriever
        self.downloader = downloader
        self.parser = parser
        self.chunker = chunker
        self.reranker = reranker

    def search(
        self,
        query: str,
        top_k: int = 20,
        top_n: int = 10,
        pdf_dir: Path = Path("data/pdfs"),
    ) -> list[Chunk]:
        papers = self.retriever.search(query, top_k=top_k)

        all_chunks = []

        for paper in papers:
            pdf_path = self.downloader.download(
                paper,
                output_dir=pdf_dir,
            )

            text = self.parser.parse(pdf_path)

            chunks = self.chunker.split(
                paper,
                text,
            )

            all_chunks.extend(chunks)

        return self.reranker.rerank(
            query,
            all_chunks,
            top_n=top_n,
        )
