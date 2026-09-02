# Process raw JSON file of articles' metadata to retrieve significant properties

from paths import *
import polars as pl

RAW_JSON_PATH = DATA_PATH / "arxiv-metadata-oai-snapshot.json"
COLS_TO_USE = ["id", "title", "abstract", "update_date"]

(
    pl.scan_ndjson(RAW_JSON_PATH)
    .select(COLS_TO_USE)
    .sink_parquet(DATA_PATH / "arxiv_metadata.parquet")
)
