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

        return scored_chunks[:top_n]
