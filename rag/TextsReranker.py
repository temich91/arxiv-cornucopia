from rag_dataclasses import *


class FullTextReranker:
    """Ranks article chunks by relevance to the query."""

    def __init__(self, reranker_model):
        self.model = reranker_model

    def rerank(
        self,
        query: str,
        chunks: list[Chunk],
        top_n: int = 10,
    ) -> list[Chunk]:
        documents = [chunk.text for chunk in chunks]
        scores = self.model.rerank(query, documents)

        scored_chunks = []

        for chunk, score in zip(chunks, scores):
            chunk.score = float(score)
            scored_chunks.append(chunk)

        scored_chunks.sort(
            key=lambda chunk: chunk.score,
            reverse=True,
        )

        best_articles_cnt = 0
        best_articles = set()
        best_articles_chunks = []
        i = 0
        while best_articles_cnt < top_n:
            if scored_chunks[i].paper_id not in best_articles:
                best_articles.add(scored_chunks[i].paper_id)
                best_articles_cnt += 1
                best_articles_chunks.append(scored_chunks[i])
            i += 1

        return best_articles_chunks
