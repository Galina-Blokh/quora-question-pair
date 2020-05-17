import os
import pandas as pd
import numpy as np
import sys


from utils.misc import timeit

from nlp.tokenizer import *


def preprocess_file_csv(input_file, output_file):
    df = pd.read_csv(input_file)
    print("Preprocess started.")
    df['preprocessed_q1'] = df['question1'].apply(preprocess_sent)
    print("Preprocess q1 finished.")
    df['preprocessed_q2'] = df['question2'].apply(preprocess_sent)
    print("Preprocess q2 finished.")
    df.to_csv(output_file)


@timeit
def preprocess_sent(sent):
    toks = (SpacyTokens(sent)
            .lower()
            .remove_all(stop_words, punct)
            .lemmatize()
            .remove_all(stop_words))
    list_of_strings = [i.text for i in toks]
    return " ".join(list_of_strings)


def normalize_sentence(q):
    return " ".join(SpacyTokens(q).lower().lemmatize().remove_all(stop_words, punct).map(lambda t: t.text))


@timeit
def preprocess_file_csv2(input_file, output_file):
    print("Preprocess started.")
    df = pd.read_csv(input_file)
    df["preprocessed_q1"] = ""
    df['preprocessed_q1'] = np.vectorize(normalize_sentence)(df['question1'].values)
    print("Preprocess q1 finished.")
    df["preprocessed_q2"] = ""
    df['preprocessed_q2'] = np.vectorize(normalize_sentence)(df['question2'].values)
    print("Preprocess q2 finished.")
    # df.to_csv(output_file)


def preprocess_sent3(sentence):
    return " ".join(SpacyTokens(sentence).lower()
                    .lemmatize()
                    .remove_all(stop_words, punct)
                    .map(lambda t: t.text))


@timeit
def preprocess_file_csv3(input_file, output_file):
    import swifter
    print("Preprocess started.")
    df = pd.read_csv(input_file)
    df["preprocessed_q1"] = ""
    df['preprocessed_q1'] = df['question1'].swifter.apply(preprocess_sent3)
    print("Preprocess q1 finished.")
    df["preprocessed_q2"] = ""
    df['preprocessed_q2'] = df['question2'].swifter.apply(preprocess_sent3)
    print("Preprocess q2 finished.")
    df.to_csv(output_file)


def preprocess_sent4(col):
    for sentence in SpacyTokens(col):
        yield list(sentence.lower()
                   .lemmatize()
                   .remove_all(stop_words, punct)
                   .map(lambda t: t.text))


@timeit
def preprocess_file_csv4(input_file, output_file):
    print("Preprocess started.")
    df = pd.read_csv(input_file)
    df["preprocessed_q1"] = ""
    df['preprocessed_q1'] = list(preprocess_sent4(df['question1'].values))
    print("Preprocess q1 finished.")
    df["preprocessed_q2"] = ""
    df['preprocessed_q2'] = list(preprocess_sent4(df['question2'].values))
    print("Preprocess q2 finished.")
    df.to_csv(output_file)


if __name__ == '__main__':
    # df = pd.read_csv("../notebooks/maria/train_dup.csv")[:100]
    preprocess_file_csv4("../notebooks/maria/train_dup.csv", "preprocess2500.csv")
    # preprocess_file_csv("../notebooks/maria/train_dup.csv", "preprocess2500.csv")
