from typing import Iterable

import spacy
from spacy.lang.en.stop_words import STOP_WORDS as stop_words
from spacy.lang.en import English
from string import punctuation as punctuations
from functools import lru_cache
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from spacy.util import compile_infix_regex
from textacy import Corpus
from textacy import vsm

nlp = None


@lru_cache(10)
def nlp_parser():
    global nlp
    if nlp is None:
        nlp = English()
        # nlp.add_pipe()
        infixes = nlp.Defaults.prefixes + tuple([r"[-]~"])
        infix_re = spacy.util.compile_infix_regex(infixes)
        nlp.tokenizer = spacy.tokenizer.Tokenizer(nlp.vocab, infix_finditer=infix_re.finditer)
        nlp.max_length = 2_000_000
    return nlp


def spacy_tokenizer(sentence: str) -> Iterable:
    parser = nlp_parser()
    tokens = (word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in parser(sentence))
    return (word for word in tokens if word not in stop_words and word not in punctuations)


class TfidfVectorizeEx(TfidfVectorizer):
    def build_analyzer(self):
        stop_words = list(self.get_stop_words())
        stop_words.extend(stop_words)

        def analyser(doc):
            return (self._word_ngrams(spacy_tokenizer(doc), stop_words))

        return (analyser)

    @staticmethod
    def from_corpus(corpus, max_features=5000, max_df=1.0, min_df=1):
        v = TfidfVectorizeEx(input='content',
                             encoding='utf-8', decode_error='replace', strip_accents='unicode',
                             lowercase=True, analyzer='word', stop_words='english',
                             token_pattern=r'(?u)\b[a-zA-Z_][a-zA-Z0-9_-]+\b',
                             ngram_range=(1, 2),
                             max_features=max_features,
                             norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=True
                             , max_df=max_df, min_df=min_df
                             )
        v.fit(corpus)
        return v


if __name__ == '__main__':
    sentence = "How does the Surface-Pro himself 4 compare with iPad Pro?"
    tokens = [token for token in spacy_tokenizer(sentence)]
    print(tokens)
    df = pd.read_csv(
        "../notebooks/maria/train_dup.csv").drop_duplicates().dropna()
    corpus = pd.concat([df['question1'], df['question2']]).unique()
    v = TfidfVectorizeEx.from_corpus(corpus)
    trainq1_trans = v.transform(df['question1'].values)
    trainq2_trans = v.transform(df['question2'].values)
