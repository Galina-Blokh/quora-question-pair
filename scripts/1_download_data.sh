#! /bin/bash
PROJECT_DIR=..
cd $PROJECT_DIR/data/raw
kaggle competitions download -c quora-question-pairs
FILE=quora-question-pairs.zip
if [ -f "$FILE" ]; then
    unzip -o $FILE -d .
    rm $FILE
fi