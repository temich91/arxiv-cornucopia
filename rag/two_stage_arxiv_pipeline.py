"""
    query
    ArxivRetriever
    Qdrant: top-K papers
    FullTextDownloader
    PDFParser
    TextChunker
    FullTextReranker
    top-N relevant chunks
    answer
"""

from dataclasses import dataclass
from ArxivRetriever import ArxivRetriever
from FullTextDownloader import FullTextDownloader
from PDFParser import PDFParser
from TextChunker import TextChunker
from TextsReranker import FullTextReranker
from Pipeline import RAGPipeline
from pathlib import Path
from rag_dataclasses import *
from utils.paths import *
from qdrant_client import QdrantClient
from fastembed import TextEmbedding
import certifi
import os
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
os.environ["SSL_CERT_FILE"] = certifi.where()
from arxiv import Client

COLLECTION_NAME = "arXiv_abstracts"
QDRANT_URL = "http://localhost:6333"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
PDF_PATH = ROOT / "temp_pdf_papers"

if __name__ == "__main__":
    client_arxiv = Client()
    client_qdrant = QdrantClient(url=QDRANT_URL, prefer_grpc=True)
    embded_model = TextEmbedding(MODEL_NAME)

    retriever = ArxivRetriever(qdrant_client=client_qdrant, embedding_model=embded_model, collection_name=COLLECTION_NAME)
    results = retriever.search(query="Aerospace", top_k=5)

    downloader = FullTextDownloader()
    p = downloader.download(arxiv_client=client_arxiv, paper=results[0], output_dir=PDF_PATH)
    parser = PDFParser()
    p_text = parser.parse(p)

    chunker = TextChunker(chunk_size=800, overlap=100)
    c = chunker.split(paper=results[0], text=p_text)


    # reranker = FullTextReranker(...)
    #
    # pipeline = RAGPipeline(
    #     retriever=retriever,
    #     downloader=downloader,
    #     parser=parser,
    #     chunker=chunker,
    #     reranker=reranker,
    # )
    #
    # results = pipeline.search(
    #     query="transformers for image classification",
    #     top_k=20,
    #     top_n=10,
    # )
    #
    # for chunk in results:
    #     print(chunk.score, chunk.title)
    #     print(chunk.text[:500])
    #
