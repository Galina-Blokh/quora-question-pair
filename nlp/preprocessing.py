from typing import Iterable
from spacy.lang.en.stop_words import STOP_WORDS as stop_words
from spacy.lang.en import English
import string
from functools import lru_cache

punctuations = string.punctuation

nlp = None

@lru_cache(10)
def nlp_parser():
    global nlp
    if nlp is None:
        nlp = English()
        # nlp.add_pipe()
        nlp.max_length = 2_000_000
    return nlp


def spacy_tokenizer(sentence: str) -> Iterable:
    parser = nlp_parser()
    tokens = (word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in parser(sentence))
    return (word for word in tokens if word not in stop_words and word not in punctuations)


if __name__ == '__main__':
    tokens = [token for token in spacy_tokenizer("How does the Surface Pro himself 4 compare with iPad Pro?")]
    print(tokens)
