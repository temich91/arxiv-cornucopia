from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = ROOT / "data"
RAW_JSON_PATH = DATA_PATH / "arxiv-metadata-oai-snapshot.json"
