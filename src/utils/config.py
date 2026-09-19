from utils.paths import DATA_PATH

COLLECTION_NAME = "arXiv_abstracts"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER_NAME = "Xenova/ms-marco-MiniLM-L-6-v2"
PDF_PATH = DATA_PATH / "temp_pdf_papers"
METADATA_PATH = DATA_PATH / "arxiv_metadata.parquet"
KAGGLE_DATASET = "Cornell-University/arxiv"