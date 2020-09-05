#! /bin/bash
PROJECT_DIR=..
cd $PROJECT_DIR/data/external || exit
MODEL_FILE=glove.840B.300d.txt
if [ ! -f "$MODEL_FILE" ]; then
    wget http://www-nlp.stanford.edu/data/glove.840B.300d.zip -O glove.840B.300d.zip
    unzip glove.840B.300d.zip
    rm glove.840B.300d.zip
fi