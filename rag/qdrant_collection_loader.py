from qdrant_client import QdrantClient
from qdrant_client.models import Document, Distance, VectorParams, PointStruct
from utils.paths import *
import polars as pl
from tqdm import tqdm

COLLECTION_NAME = "arXiv_abstracts"
QDRANT_URL = "http://localhost:6333"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 256
METADATA_PATH = DATA_PATH / "arxiv_metadata.parquet"

class CollectionLoader:
    def __init__(self, collection_name=COLLECTION_NAME, metadata_path=METADATA_PATH):
        self.client = QdrantClient(url=QDRANT_URL)
        self.metadata = self._get_metadata(metadata_path)
        self.collection_name = collection_name
        self.create_collection()
        self.fill_collection()

    def _get_metadata(self, path):
        return pl.read_parquet(path)

    def create_collection(self):
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.client.get_embedding_size(MODEL_NAME),
                                            distance=Distance.COSINE)
            )

    def fill_collection(self, batch_size=BATCH_SIZE):
        for batch in tqdm(self.metadata.iter_slices(n_rows=batch_size), desc="Batches processed"):
            documents = [Document(text=text, model=MODEL_NAME) for text in batch["abstract"].to_list()]
            payloads = batch.select(["id", "title", "update_date"]).to_dicts()

            self.client.upload_collection(
                collection_name=self.collection_name,
                vectors=documents,
                payload=payloads,
                batch_size=batch_size
            )

    def get_collection_info(self):
        return self.client.get_collection(collection_name=self.collection_name)

    def search_collection(self, query_text, limit=5):
        return self.client.query_points(
            collection_name=self.collection_name,
            query=Document(text=query_text, model=MODEL_NAME),
            limit=limit
        )

loader = CollectionLoader()
print(loader.search_collection(query_text="RAG"))



