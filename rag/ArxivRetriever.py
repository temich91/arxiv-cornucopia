from rag_dataclasses import *
import arxiv


class ArxivRetriever:
    """Searches the existing Qdrant collection using embeddings."""

    def __init__(self, qdrant_client, embedding_model, collection_name: str):
        self.client = qdrant_client
        self.model = embedding_model
        self.collection_name = collection_name

    def _get_pdf_url(self, arxiv_client: arxiv.Client, arxiv_id: str) -> str:
        search = arxiv.Search(id_list=[arxiv_id])
        return next(arxiv_client.results(search)).pdf_url

    def search(self, arxiv_client: arxiv.Client, query: str, top_k: int = 20) -> list[Paper]:
        """Return the top-K papers from Qdrant.

        The actual conversion from Qdrant payload -> Paper depends on
        the payload schema used during ingestion.
        """
        query_vector = list(self.model.embed(query))[0].tolist()

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True
        ).points

        papers = []

        for result in results:
            payload = result.payload

            papers.append(
                Paper(
                    arxiv_id=payload["id"],
                    title=payload["title"],
                    abstract=payload["abstract"],
                    update_date=payload["update_date"],
                    pdf_url=self._get_pdf_url(arxiv_client, payload["id"])
                )
            )

        return papers
