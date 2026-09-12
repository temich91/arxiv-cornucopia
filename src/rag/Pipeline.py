from rag.ArxivRetriever import ArxivRetriever
from rag.FullTextDownloader import FullTextDownloader
from rag.PDFParser import PDFParser
from rag.TextChunker import TextChunker
from rag.TextsReranker import FullTextReranker
from pathlib import Path
from arxiv import Client
from dataclasses import asdict
import json
import os
import time
from utils.constants import *


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
        pdf_path: Path
    ):
        self.arxiv_client = arxiv_client
        self.retriever = retriever
        self.downloader = downloader
        self.parser = parser
        self.chunker = chunker
        self.reranker = reranker
        self.pdf_path = pdf_path
        if not os.path.exists(self.pdf_path):
            os.mkdir(self.pdf_path)

    def search(
        self,
        query: str,
        candidates_cnt: int = 10,
        top_chunks_cnt: int = 5,
    ) -> list[str]:
        papers = self.retriever.search(self.arxiv_client, query, top_k=candidates_cnt)
        print(len(papers))
        texts = []

        for i in range(len(papers)):
            paper = self.downloader.download(paper=papers[i], output_dir=self.pdf_path)
            time.sleep(PDF_DOWNLOAD_TIMEOUT)
            paper_text = self.parser.parse(paper)
            texts.append(paper_text)

        all_chunks = [chunk for i in range(len(papers))
                      for chunk in self.chunker.split(paper=papers[i], text=texts[i])]

        self.downloader.clean(self.pdf_path)

        ranked_chunks = self.reranker.rerank(
            query=query,
            chunks=all_chunks,
            top_n=top_chunks_cnt,
        )
        return [json.dumps(asdict(chunk)) for chunk in ranked_chunks]
