from rag.ArxivRetriever import ArxivRetriever
from rag.FullTextDownloader import FullTextDownloader
from rag.PDFParser import PDFParser
from rag.TextChunker import TextChunker
from rag.TextsReranker import FullTextReranker
from rag.Pipeline import RAGPipeline
from arxiv import Client
from qdrant_client import QdrantClient
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from utils.paths import *

COLLECTION_NAME = "arXiv_abstracts"
QDRANT_URL = "http://localhost:6333"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER_NAME = "Xenova/ms-marco-MiniLM-L-6-v2"
PDF_PATH = DATA_PATH / "temp_pdf_papers"


def create_pipeline():
    client_arxiv = Client()
    client_qdrant = QdrantClient(url=QDRANT_URL, prefer_grpc=True)
    embdedding_model = TextEmbedding(MODEL_NAME)
    retriever = ArxivRetriever(qdrant_client=client_qdrant,
                               embedding_model=embdedding_model,
                               collection_name=COLLECTION_NAME)
    downloader = FullTextDownloader()
    parser = PDFParser()
    chunker = TextChunker(chunk_size=800, overlap=100)
    rerank_model = TextCrossEncoder(CROSS_ENCODER_NAME)
    reranker = FullTextReranker(reranker_model=rerank_model)

    return RAGPipeline(
        arxiv_client=client_arxiv,
        retriever=retriever,
        downloader=downloader,
        parser=parser,
        chunker=chunker,
        reranker=reranker,
        pdf_path=PDF_PATH
    )
