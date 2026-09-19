# Collecting weekly dump of metadata of all arXiv papers from Kaggle.
# Requires Kaggle API key in "~/.kaggle/access_token"
# Alternative: download latest release from https://www.kaggle.com/datasets/Cornell-University/arxiv
# to /data/arxiv-metadata-oai-snapshot.json

import kaggle
from paths import DATA_PATH
from utils.config import KAGGLE_DATASET
from utils.paths import ROOT
import sys

sys.path.append(ROOT)
kaggle.api.authenticate()
kaggle.api.dataset_download_files(KAGGLE_DATASET, path=DATA_PATH, unzip=True)
