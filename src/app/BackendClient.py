from rag.Pipeline import RAGPipeline
import json
import polars as pl


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
