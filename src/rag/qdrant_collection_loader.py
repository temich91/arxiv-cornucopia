from qdrant_client import QdrantClient
from qdrant_client.models import Batch, Distance, VectorParams
from src.utils.paths import *
import polars as pl
from tqdm import tqdm
from fastembed import TextEmbedding
from uuid6 import uuid7
from utils.constants import *
from utils.config import *


class CollectionLoader:
    def __init__(self, collection_name=COLLECTION_NAME, metadata_path=METADATA_PATH):
        self.client = QdrantClient(url=QDRANT_URL, prefer_grpc=True)
        self.model = TextEmbedding(model_name=MODEL_NAME)
        self.metadata = self._get_metadata(metadata_path)
        self.collection_name = collection_name
        self.create_collection()
        self.fill_collection()

    def _get_metadata(self, path):
        return pl.read_parquet(path, n_rows=METADATA_ROWS_TO_LOAD)

    def create_collection(self):
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.client.get_embedding_size(MODEL_NAME),
                                            distance=Distance.COSINE)
            )

    def fill_collection(self, batch_size=BATCH_SIZE):
        for batch in tqdm(self.metadata.iter_slices(n_rows=batch_size), desc="Batches processed"):
            texts = batch["abstract"].to_list()
            payloads = batch.select(["id", "title", "abstract", "update_date"]).to_dicts()
            embeddings = self.model.embed(texts)
            ids = [uuid7().hex for _ in range(len(texts))]

            self.client.upsert(
                collection_name=self.collection_name,
                points=Batch(
                    ids=ids,
                    vectors=embeddings,
                    payloads=payloads
                )
            )

    def get_collection_info(self):
        return self.client.get_collection(collection_name=self.collection_name)

    def search_collection(self, query_text, limit=5):
        query_vector = list(self.model.embed(query_text))[0].tolist()
        return self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit
        )

loader = CollectionLoader()
