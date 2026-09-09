import streamlit as st
import polars as pl
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
import json
from fastembed.rerank.cross_encoder import TextCrossEncoder
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
os.environ["SSL_CERT_FILE"] = certifi.where()
from arxiv import Client

class BackendClient:
    """Клиент для отправки поискового запроса и получения чанков ответов."""

    def __init__(self, pipeline: RAGPipeline):
        self.pipeline = pipeline

    def fetch_results(self, query: str, max_results: int) -> pl.DataFrame:
        """Отправляет запрос и возвращает Polars DataFrame с результатами."""
        papers = self.pipeline.search(
            query=query,
            candidates_cnt=max_results * 2,
            top_chunks_cnt=max_results
        )

        dicts_list = [json.loads(paper) for paper in papers]

        return pl.DataFrame(dicts_list)


class StreamlitApp:
    """Класс для управления интерфейсом Streamlit."""

    def __init__(self, pipeline):
        self.client = BackendClient(pipeline)

    def run(self):
        st.set_page_config(
            page_title="ArXiv RAG UI",
            page_icon="📚",
            layout="wide"
        )
        st.title("📚 ArXiv Search")
        st.markdown("Интерфейс поиска по статьям.")

        query, max_results, search_clicked = self._render_sidebar()

        if search_clicked:
            with st.spinner("Выполняется поиск и сбор чанков..."):
                try:
                    df_results = self.client.fetch_results(query, max_results)
                    st.session_state["search_results"] = df_results
                except Exception as e:
                    st.error(f"Ошибка при выполнении запроса: {e}")

        self._render_results()

    def _render_sidebar(self):
        st.sidebar.header("Параметры поиска")
        query = st.sidebar.text_input("Поисковый запрос", "Retrieval-Augmented Generation")
        max_results = st.sidebar.slider("Максимум результатов", min_value=1, max_value=20, value=5, step=1)
        search_clicked = st.sidebar.button("Искать")
        return query, max_results, search_clicked

    def _render_results(self):
        if "search_results" in st.session_state:
            df = st.session_state["search_results"]

            if not df.is_empty():
                st.success(f"Найдено документов: {len(df)}")

                st.subheader("Список документов")
                st.dataframe(
                    df.select(["title", "update_date", "pdf_url"]),
                    use_container_width=True
                )

                st.markdown("---")
                st.subheader("Отрывки по теме")

                for row in df.to_dicts():
                    with st.expander(f"{row['title']} ({row['update_date']})"):
                        st.markdown(f"**Дата публикации:** {row['update_date']}")
                        st.markdown(f"[Скачать PDF]({row['pdf_url']})")
                        st.markdown("---")
                        st.markdown("**Фрагмент:**")
                        st.info(row['text'])
            else:
                st.warning("По вашему запросу ничего не найдено.")
        else:
            st.info("Введите запрос в боковой панели и нажмите **Искать**.")


COLLECTION_NAME = "arXiv_abstracts"
QDRANT_URL = "http://localhost:6333"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER_NAME = "Xenova/ms-marco-MiniLM-L-6-v2"
PDF_PATH = DATA_PATH / "temp_pdf_papers"

if __name__ == "__main__":
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

    query = "dark matter model"

    pipeline = RAGPipeline(
        arxiv_client=client_arxiv,
        retriever=retriever,
        downloader=downloader,
        parser=parser,
        chunker=chunker,
        reranker=reranker,
        pdf_path=PDF_PATH
    )
    app = StreamlitApp(pipeline)
    app.run()
