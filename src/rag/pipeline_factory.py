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
from utils.constants import *
from utils.config import *
import os

host = os.getenv("QDRANT_HOST", "127.0.0.1")
port = int(os.getenv("QDRANT_PORT", 6333))

client = QdrantClient(host=host, port=port)
def create_pipeline():
    client_arxiv = Client()
    client_qdrant = QdrantClient(url=host, port=port, prefer_grpc=True)
    embdedding_model = TextEmbedding(MODEL_NAME)
    retriever = ArxivRetriever(qdrant_client=client_qdrant,
                               embedding_model=embdedding_model,
                               collection_name=COLLECTION_NAME)
    downloader = FullTextDownloader()
    parser = PDFParser()
    chunker = TextChunker(chunk_size=CHUNK_SIZE, overlap=OVERLAP)
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
