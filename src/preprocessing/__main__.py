import numpy as np
import pandas as pd
import contractions
import emoji
import os
import json
from definitions import *
from string import punctuation
from itertools import groupby

try:
    import regex as re
except:
    import re

from functools import lru_cache


def clean(in_data: pd.DataFrame) -> pd.DataFrame:
    assert 'question1' in in_data.columns
    assert 'question2' in in_data.columns

    print("removing nan")
    in_data = in_data[~in_data.isnull()]
    in_data = in_data[(~in_data["question1"].isna()) & (~in_data["question2"].isna())]

    print("fixing contractions")
    in_data['question1'] = np.vectorize(contractions.fix)(in_data['question1'])
    in_data['question2'] = np.vectorize(contractions.fix)(in_data['question2'])

    print("fixing emoji")
    in_data['question1'] = np.vectorize(emoji.demojize)(in_data['question1'])
    in_data['question2'] = np.vectorize(emoji.demojize)(in_data['question2'])

    print("cleaning")
    in_data['question1'] = clean_sentence(in_data['question1'])
    in_data['question2'] = clean_sentence(in_data['question2'])

    in_data = in_data[~in_data.isnull()]
    in_data = in_data[(~in_data["question1"].isna()) & (~in_data["question2"].isna())]

    in_data['question1'] = in_data['question1'].str.lower()
    in_data['question2'] = in_data['question2'].str.lower()
    return in_data


@lru_cache(1)
def get_currencies():
    currencies = json.load(open(os.path.join(PROJECT_DIR, "data", "external", "currencies.json"), "r"))
    return currencies


def clean_sentence(col: pd.Series) -> pd.Series:
    col = col.str.strip()
    col = col.str.replace('\n', ' ')
    col = col.str.replace('’', '\'')
    col = col.str.replace('é', 'e')
    col = col.str.replace('“', '"')
    col = col.str.replace('″', '"')
    col = col.str.replace('«', '"')
    col = col.str.replace('»', '"')
    col = col.str.replace('”', '"')
    col = col.str.replace('`', '\'')
    col = col.str.replace('′', '\'')
    col = col.str.replace('‘', '\'')
    col = col.str.replace('…', '...')
    col = col.str.replace('－', '-')
    col = col.str.replace('–', '-')
    col = col.str.replace('×', '*')
    col = col.str.replace('\u202a', ' ')
    col = col.str.replace('\u202c', ' ')
    col = col.str.replace('\u202e', ' ')
    col = col.str.replace('\u200b', ' ')
    col = col.str.replace('\u200f', ' ')
    col = col.str.replace('\x7f', ' ')
    col = col.str.replace('？', '?')
    col = col.str.replace('Î', 'I')
    col = col.str.replace('à', 'a')
    col = col.str.replace('á', 'a')
    col = col.str.replace('ã', 'a')
    col = col.str.replace('í', 'i')
    col = col.str.replace('ó', 'o')
    col = col.str.replace('ö', 'o')
    col = col.str.replace('÷', '\/')
    col = col.str.replace('ü', 'u')
    col = col.str.replace('ı', '1')
    col = col.str.replace('ṭ', 't')
    currencies = get_currencies()
    for k, v in currencies.items():
        col = col.str.replace(k, ' ' + v + ' ')
    col = col.str.replace(r"[^A-Za-z0-9(),!.?\'\`]", " ")
    col = col.str.replace(r",", " ")
    col = col.str.replace(r"\.", " ")
    col = col.str.replace(r"!", " ")
    col = col.str.replace(r"\(", " ( ")
    col = col.str.replace(r"\)", " ) ")
    col = col.str.replace(r"\?", " ")
    col = col.apply(lambda s: " ".join(s.split()))
    return col

def preprocessing_data(in_file: str, out_file: str):
    df = clean(pd.read_csv(in_file))
    if os.path.isdir(out_file):
        out_file = os.path.join(out_file, os.path.basename(in_file))
    df.to_csv(out_file, index=False)


if __name__ == '__main__':
    import fire

    fire.Fire(preprocessing_data)
