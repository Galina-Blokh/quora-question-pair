import os
from functools import lru_cache
from typing import Callable, List

import numpy as np
import pandas as pd
import spacy
from fuzzywuzzy import fuzz
from spacy.lang.en import English
from spacy.language import Language
from spacy.tokenizer import Tokenizer

from src.nlp.stop_words import STOP_WORDS, QUESTION_WORDS

__questions__ = {"who": 1 << 1, "whom": 1 << 2, "whose": 1 << 3, "what": 1 << 4, "when": 1 << 5, "where": 1 << 6,
                 "why": 1 << 7, "how": 1 << 8,
                 "there": 1 << 9, "that": 1 << 10, "which": 1 << 11, "whither": 1 << 12, "whence": 1 << 13,
                 "whether": 1 << 14, "whatsoever": 1 << 15}


@lru_cache(10)
def nlp_parser(name="en_core_web_sm") -> Language:
    global nlp
    if nlp is None:
        try:
            nlp = spacy.load(name)
        except:
            nlp = English()
        nlp.tokenizer = create_tokenizer(nlp)
        nlp.max_length = 2_000_000
    return nlp

@lru_cache(10)
def create_tokenizer(nlp):
    cls = nlp.Defaults
    nlp.Defaults.stop_words |= STOP_WORDS
    rules = cls.tokenizer_exceptions
    token_match = cls.token_match
    prefix_search = (spacy.util.compile_prefix_regex(cls.prefixes).search if cls.prefixes else None)
    suffix_search = (spacy.util.compile_suffix_regex(cls.suffixes).search if cls.suffixes else None)
    infixes = cls.prefixes + tuple([x for x in nlp.Defaults.infixes if '-|–|—|--|---|——|~' not in x])
    infix_finditer = (spacy.util.compile_infix_regex(infixes).finditer if infixes else None)
    vocab = nlp.vocab if nlp is not None else cls.create_vocab(nlp)
    for w in QUESTION_WORDS:
        vocab[w.lower()].is_stop = False
        vocab[w.title()].is_stop = False
        vocab[w.upper()].is_stop = False
    return Tokenizer(
        vocab,
        rules=rules,
        prefix_search=prefix_search,
        suffix_search=suffix_search,
        infix_finditer=infix_finditer,
        token_match=token_match,
    )

def count_tokens(tokenizer: Callable[[str], List], in_array: np.array) -> np.array:
    return np.fromiter((len(tokenizer(s)) for s in in_array), dtype=np.int16)


def count_without_punct_stop_words(tokenizer: Callable[[str], List], in_array: np.array) -> np.array:
    return np.fromiter(
        (
            sum(
                1
                for t in tokenizer(s)
                if not t.is_punct and t.text not in STOP_WORDS
            )
            for s in in_array
        ),
        dtype=np.int16,
    )


def get_questions_mask(tokenizer: Callable[[str], List], in_array: np.array) -> np.array:
    return np.fromiter((sum(__questions__.get(t.text.lower(), 0) for t in tokenizer(s)) for s in in_array),
                       dtype=np.int16)


def create_wh_ds(df:pd.DataFrame, target_column:str, out_column:str)->pd.DataFrame:
    dfw = df[[target_column]][:].reset_index(drop=True)
    dfw[out_column] = get_questions_mask(dfw[target_column].values)
    for q, mask in __questions__.items():
        dfw[q] = (np.bitwise_and(dfw[out_column], mask) != 0).astype(int)
    return dfw


def common_tokens_count(tokenizer: Callable[[str], List], q1: str, q2: str) -> int:
    q1_tokens = {t.lemma_ for t in tokenizer(q1)}
    q2_tokens = {t.lemma_ for t in tokenizer(q2)}
    return len(q1_tokens.intersection(q2_tokens))

def building_features(in_data: pd.DataFrame, tokenizer) -> pd.DataFrame:
    assert "question1" in in_data.columns
    assert "question2" in in_data.columns
    nlp = nlp_parser()
    tokenizer = nlp.tokenizer

    in_data["len_char1"] = in_data["question1"].str.len()
    in_data["len_char2"] = in_data["question2"].str.len()
    in_data["tokens_count1"] = count_tokens(tokenizer, in_data['question1'].values)
    in_data["tokens_count2"] = count_tokens(tokenizer, in_data['question2'].values)
    in_data["tokens_wps_count1"] = count_without_punct_stop_words(tokenizer, in_data['question1'].values)
    in_data["tokens_wps_count2"] = count_without_punct_stop_words(tokenizer, in_data['question2'].values)
    in_data["questions_mask1"] = get_questions_mask(tokenizer, in_data['question1'].values)
    in_data["questions_mask2"] = get_questions_mask(tokenizer, in_data['question2'].values)
    in_data["token_sort_ratio"] = np.vectorize(fuzz.token_sort_ratio)(in_data['question1'], in_data['question2'])
    in_data["token_set_ratio"] = np.vectorize(fuzz.token_set_ratio)(in_data['question1'], in_data['question2'])
    in_data["wratio"] = np.vectorize(fuzz.WRatio)(in_data['question1'], in_data['question2'])
    in_data["common_tokens_count"] = np.vectorize(common_tokens_count)(in_data['question1'], in_data['question2'])
    in_data["len_lat_char1"] = in_data["question1"].replace({'([^\x00-\x7A])+': ''}, regex=True).str.strip().str.len()
    in_data["len_lat_char2"] = in_data["question2"].replace({'([^\x00-\x7A])+': ''}, regex=True).str.strip().str.len()
    return in_data

def create_features(in_file:str, out_file:str):
    df = building_features(pd.read_csv(in_file))
    if os.path.isdir(out_file):
        out_file = os.path.join(out_file, os.path.basename(in_file))
    df.to_csv(out_file, index=False)

if __name__ == '__main__':
    import fire

    fire.Fire(create_features)
