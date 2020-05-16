import os
import pandas as pd
import numpy as np
import sys

sys.path.append(os.path.abspath('../../'))
from nlp.tokenizer import *


def preprocess_file_csv(input_file, output_file):
    df = pd.read_csv(input_file)
    print("Preprocess started.")
    df['preprocessed_q1'] = df['question1'].apply(preprocess_sent)
    print("Preprocess q1 finished.")
    df['preprocessed_q2'] = df['question2'].apply(preprocess_sent)
    print("Preprocess q2 finished.")
    df.to_csv(output_file)


def preprocess_sent(sent):
    toks = SpacyTokens(sent).lower().remove_all(stop_words).remove_all(punct).\
        lemmatize().remove_all(stop_words)
    list_of_strings = [i.text for i in toks]
    return " ".join(list_of_strings)


if __name__ == '__main__':
    preprocess_file_csv("../notebooks/maria/train_dup.csv", "preprocess2500.csv")