from dataclasses import dataclass

@dataclass
class Paper:
    arxiv_id: str
    title: str
    abstract: str
    update_date: str

@dataclass
class Chunk:
    paper_id: str
    title: str
    text: str
    score: float | None = None
