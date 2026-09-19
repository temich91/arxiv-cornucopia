from app.BackendClient import BackendClient
import streamlit as st
from utils.constants import *

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
                df_results = self.client.fetch_results(
                    query,
                    max_results,
                )
                st.session_state["search_results"] = df_results


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
