#! /bin/bash
PROJECT_DIR=..
python -m src.features.build_features $PROJECT_DIR/data/interim/train.csv $PROJECT_DIR/data/processed/train.csv