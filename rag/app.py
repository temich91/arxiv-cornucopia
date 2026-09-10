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

MIN_RESULTS = 1
MAX_RESULTS = 20

class StreamlitApp:
    """Streamlit web interface class."""

    def __init__(self, pipeline):
        self.client = BackendClient(pipeline)

    def run(self):
        st.set_page_config(
            page_title="ArXiv Search",
            page_icon="🔍",
            layout="wide",
        )

        st.title("ArXiv Search")
        st.caption("Семантический поиск научных публикаций.")

        with st.container():
            col_query, col_slider, col_btn = st.columns([3, 1.5, 1])

            with col_query:
                query = st.text_input(
                    "Поисковый запрос",
                    "Graph",
                )

            with col_slider:
                max_results = st.slider(
                    "Максимум результатов",
                    min_value=MIN_RESULTS,
                    max_value=MAX_RESULTS,
                    value=5,
                    step=1,
                )

            with col_btn:
                search_clicked = st.button(
                    "Запустить поиск",
                    use_container_width=True,
                )

        st.divider()

        if search_clicked:
            with st.spinner(
                "Синхронизация с базой ArXiv и обработка эмбеддингов..."
            ):
                try:
                    df_results = self.client.fetch_results(
                        query,
                        max_results,
                    )
                    st.session_state["search_results"] = df_results

                except Exception as e:
                    st.error(
                        f"Произошла ошибка при обработке запроса: {e}"
                    )

        self._render_results()

    def _render_results(self):
        if "search_results" not in st.session_state:
            st.info(
                "### Начните исследование\n"
                "Введите поисковый запрос в поле выше "
                "и нажмите кнопку запуска."
            )
            return

        df = st.session_state["search_results"]

        if df.is_empty():
            st.warning(
                "По вашему запросу не удалось обнаружить "
                "релевантных материалов."
            )
            return

        st.success(f"Найдено документов: {len(df)}")

        st.subheader("Сводка документов")

        st.dataframe(
            df.select(["title", "update_date", "pdf_url"]),
            use_container_width=True,
            hide_index=True,
        )

        st.divider()

        st.subheader("Извлеченные фрагменты текста")

        for row in df.to_dicts():
            with st.expander(
                f"{row['title']}  ({row['update_date']})"
            ):
                col1, col2 = st.columns([3, 1])

                with col1:
                    st.caption(
                        f"Дата обновления: {row['update_date']}"
                    )

                with col2:
                    st.markdown(
                        f"[Открыть PDF]({row['pdf_url']})"
                    )

                st.divider()

                st.caption("НАЙДЕННЫЙ КОНТЕКСТ")

                st.info(row["text"])



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
