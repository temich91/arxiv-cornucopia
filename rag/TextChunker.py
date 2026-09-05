from rag_dataclasses import *

class TextChunker:
    """Splits article text into chunks suitable for reranking/RAG."""

    def __init__(self, chunk_size: int = 800, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, paper: Paper, text: str) -> list[Chunk]:
        """Split one article into overlapping chunks."""
        chunks = []

        start = 0

        while start < len(text):
            end = start + self.chunk_size

            chunks.append(
                Chunk(
                    paper_id=paper.arxiv_id,
                    title=paper.title,
                    text=text[start:end],
                )
            )

            start += self.chunk_size - self.overlap

        return chunks
