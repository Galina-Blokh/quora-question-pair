import numpy as np
import pandas as pd
import contractions
import emoji
import os
import json
from definitions import *
try:
    import regex as re
except:
    import re

from functools import lru_cache


def clean(in_data: pd.DataFrame) -> pd.DataFrame:
    assert 'question1' in in_data.columns
    assert 'question2' in in_data.columns

    print("filling nan")
    in_data['question1'] = in_data['question1'].fillna("")
    in_data['question2'] = in_data['question2'].fillna("")

    print("fixing contractions")
    in_data['question1'] = np.vectorize(contractions.fix)(in_data['question1'])
    in_data['question2'] = np.vectorize(contractions.fix)(in_data['question2'])

    print("fixing emoji")
    in_data['question1'] = np.vectorize(emoji.demojize)(in_data['question1'])
    in_data['question2'] = np.vectorize(emoji.demojize)(in_data['question2'])

    print("cleaning")
    in_data['question1'] = np.vectorize(clean_sentence)(in_data['question1'])
    in_data['question2'] = np.vectorize(clean_sentence)(in_data['question2'])
    return in_data


@lru_cache(1)
def get_currencies():
    currencies = json.load(open(os.path.join(PROJECT_DIR, "data", "external", "currencies.json"), "r"))
    return currencies


def clean_sentence(string: str) -> str:
    string = re.sub('\n', ' ', string)
    string = re.sub('’', '\'', string)
    string = re.sub('é', 'e', string)
    string = re.sub('“', '"', string)
    string = re.sub('″', '"', string)
    string = re.sub('«', '"', string)
    string = re.sub('»', '"', string)
    string = re.sub('”', '"', string)
    string = re.sub('`', '\'', string)
    string = re.sub('′', '\'', string)
    string = re.sub('‘', '\'', string)
    string = re.sub('…', '...', string)
    string = re.sub('－', '-', string)
    string = re.sub('–', '-', string)
    string = re.sub('×', '*', string)
    string = re.sub('\u202a', ' ', string)
    string = re.sub('\u202c', ' ', string)
    string = re.sub('\u202e', ' ', string)
    string = re.sub('\u200b', ' ', string)
    string = re.sub('\u200f', ' ', string)
    string = re.sub('\x7f', ' ', string)
    string = re.sub('？', '?', string)
    string = re.sub('Î', 'I', string)
    string = re.sub('à', 'a', string)
    string = re.sub('á', 'a', string)
    string = re.sub('ã', 'a', string)
    string = re.sub('í', 'i', string)
    string = re.sub('ó', 'o', string)
    string = re.sub('ö', 'o', string)
    string = re.sub('÷', '\/', string)
    string = re.sub('ü', 'u', string)
    string = re.sub('ı', '1', string)
    string = re.sub('ṭ', 't', string)
    currencies = get_currencies()
    for k, v in currencies.items():
        string = string.replace(k, ' ' + v + ' ')
    string = re.sub('\s{2,}', ' ', string)
    return string

def preprocessing_data(in_file: str, out_file: str):
    df = clean(pd.read_csv(in_file))
    if os.path.isdir(out_file):
        out_file = os.path.join(out_file, os.path.basename(in_file))
    df.to_csv(out_file, index=False)


if __name__ == '__main__':
    import fire
    fire.Fire(preprocessing_data)
