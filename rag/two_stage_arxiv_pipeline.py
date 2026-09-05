from ArxivRetriever import ArxivRetriever
from FullTextDownloader import FullTextDownloader
from PDFParser import PDFParser
from TextChunker import TextChunker
from TextsReranker import FullTextReranker
from Pipeline import RAGPipeline
from utils.paths import *
from qdrant_client import QdrantClient
from fastembed import TextEmbedding
import certifi
import os
from fastembed.rerank.cross_encoder import TextCrossEncoder
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
os.environ["SSL_CERT_FILE"] = certifi.where()
from arxiv import Client


COLLECTION_NAME = "arXiv_abstracts"
QDRANT_URL = "http://localhost:6333"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER_NAME = "Xenova/ms-marco-MiniLM-L-6-v2"
PDF_PATH = ROOT / "temp_pdf_papers"

if __name__ == "__main__":
    client_arxiv = Client()
    client_qdrant = QdrantClient(url=QDRANT_URL, prefer_grpc=True)
    embded_model = TextEmbedding(MODEL_NAME)
    retriever = ArxivRetriever(qdrant_client=client_qdrant, embedding_model=embded_model, collection_name=COLLECTION_NAME)
    downloader = FullTextDownloader()
    parser = PDFParser()
    chunker = TextChunker(chunk_size=800, overlap=100)
    rerank_model = TextCrossEncoder(CROSS_ENCODER_NAME)
    reranker = FullTextReranker(reranker_model=rerank_model)

    query = "dark matter model"

    pipeline = RAGPipeline(
        arxiv_client=client_arxiv,
        retriever=retriever,
        downloader=downloader,
        parser=parser,
        chunker=chunker,
        reranker=reranker
    )

    papers = pipeline.search(
        query=query,
        candidates_cnt=10,
        top_chunks_cnt=5,
        pdf_dir=PDF_PATH
    )

    for chunk in papers:
        print(chunk.score, chunk.title)
        print(chunk.text[:500])
