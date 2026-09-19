from rag.qdrant_collection_loader import CollectionLoader
from utils.config import COLLECTION_NAME, METADATA_PATH
from utils.constants import BATCH_SIZE
from utils.paths import ROOT
import sys

sys.path.append(ROOT)
if __name__ == "__main__":
    loader = CollectionLoader(collection_name=COLLECTION_NAME,
                              metadata_path=METADATA_PATH)
    loader.fill_collection(BATCH_SIZE)