#!/bin/bash
DOWNLOAD_PATH="./data/raw/"


kaggle competitions download -c quora-question-pairs -p ./data/ || echo "try install: pip install kaggle"

echo "download done!"
