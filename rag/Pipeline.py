from ArxivRetriever import ArxivRetriever
from FullTextDownloader import FullTextDownloader
from PDFParser import PDFParser
from TextChunker import TextChunker
from TextsReranker import FullTextReranker
from pathlib import Path
from arxiv import Client
from rag_dataclasses import *


class RAGPipeline:
    """Coordinates the complete two-stage retrieval pipeline."""

    def __init__(
        self,
        arxiv_client: Client,
        retriever: ArxivRetriever,
        downloader: FullTextDownloader,
        parser: PDFParser,
        chunker: TextChunker,
        reranker: FullTextReranker,
    ):
        self.arxiv_client = arxiv_client
        self.retriever = retriever
        self.downloader = downloader
        self.parser = parser
        self.chunker = chunker
        self.reranker = reranker

    def search(
        self,
        query: str,
        candidates_cnt: int = 10,
        top_chunks_cnt: int = 5,
        pdf_dir: Path = Path("data/temp_pdf_papers"),
    ) -> list[Chunk]:
        papers = self.retriever.search(query, top_k=candidates_cnt)

        texts = []

        for i in range(len(papers)):
            paper = self.downloader.download(arxiv_client=self.arxiv_client, paper=papers[i], output_dir=pdf_dir)
            paper_text = self.parser.parse(paper)
            texts.append(paper_text)
        all_chunks = [chunk for i in range(len(papers)) for chunk in self.chunker.split(paper=papers[i], text=texts[i])]

        self.downloader.clean(pdf_dir)

        return self.reranker.rerank(
            query=query,
            chunks=all_chunks,
            top_n=top_chunks_cnt,
        )
