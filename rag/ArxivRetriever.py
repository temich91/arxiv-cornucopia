from rag_dataclasses import *


class ArxivRetriever:
    """Searches the existing Qdrant collection using embeddings."""

    def __init__(self, qdrant_client, embedding_model, collection_name: str):
        self.client = qdrant_client
        self.model = embedding_model
        self.collection_name = collection_name

    def search(self, query: str, top_k: int = 20) -> list[Paper]:
        """Return the top-K papers from Qdrant.

        The actual conversion from Qdrant payload -> Paper depends on
        the payload schema used during ingestion.
        """
        query_vector = list(self.model.embed(query))[0].tolist()

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
        ).points

        papers = []

        for result in results:
            payload = result.payload

            papers.append(
                Paper(
                    arxiv_id=payload["id"],
                    title=payload["title"],
                    abstract=payload["abstract"],
                    pdf_url=payload["pdf_url"],
                )
            )

        return papers
