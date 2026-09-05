from rag_dataclasses import *
from langchain_text_splitters import RecursiveCharacterTextSplitter


class TextChunker:
    """Splits article text into chunks suitable for reranking/RAG."""

    def __init__(self, chunk_size: int = 800, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, paper: Paper, text: str) -> list[Chunk]:
        """Split one article into overlapping chunks."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

        if not text:
            return []

        text_chunks = text_splitter.split_text(text)

        return [Chunk(paper_id=paper.arxiv_id,
                      title=paper.title,
                      text=text)
                for text in text_chunks]
