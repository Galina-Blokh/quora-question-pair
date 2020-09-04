#! /bin/bash
PROJECT_DIR=..
python -m src.preprocessing $PROJECT_DIR/data/raw/train.csv.zip $PROJECT_DIR/data/interim/train.csv
python -m src.preprocessing $PROJECT_DIR/data/raw/test.csv.zip $PROJECT_DIR/data/interim/test.csv