import re
from functools import lru_cache, wraps
from itertools import chain

import spacy
from flupy.fluent import self_to_flu, Fluent, flu
from spacy.lang.en import English
from spacy.language import Language
from spacy.tokenizer import Tokenizer
from spacy.tokens import Doc, Token

import utils
from utils.pipeline import Pipeline

space = lambda t: t.is_space


def punct(t):
    return t.is_punct


# punct = lambda t: t.is_punct
stop = lambda t: t.is_stop
number = lambda t: t.like_num

nlp = None


@lru_cache(10)
def nlp_parser(name="en_core_web_md") -> Language:
    global nlp
    if nlp is None:
        nlp = spacy.load(name)
        infixes = nlp.Defaults.prefixes + tuple([r"[-]~"])
        infix_re = spacy.util.compile_infix_regex(infixes)
        nlp.tokenizer = spacy.tokenizer.Tokenizer(nlp.vocab, infix_finditer=infix_re.finditer)
        nlp.max_length = 2_000_000
    return nlp


class SpacyTokenizer(Tokenizer):

    def __init__(self, *args, **kwargs):
        nlp = args[0] if args and isinstance(args[0], Language) else English()
        super(SpacyTokenizer, self).__init__(nlp.vocab)

    def __call__(self, *args, **kwargs):
        return Pipeline(Tokenizer.__call__(self, *args, **kwargs))

    @staticmethod
    @lru_cache(10)
    def from_lang(nlp=None):
        if nlp is None:
            nlp = nlp_parser()
        cls = nlp.Defaults
        rules = cls.tokenizer_exceptions
        token_match = cls.token_match
        prefix_search = (
            spacy.util.compile_prefix_regex(cls.prefixes).search if cls.prefixes else None
        )
        suffix_search = (
            spacy.util.compile_suffix_regex(cls.suffixes).search if cls.suffixes else None
        )
        infix_finditer = (
            spacy.util.compile_infix_regex(cls.prefixes + tuple([r"[-]~"])).finditer if cls.prefixes else None
        )
        vocab = nlp.vocab if nlp is not None else cls.create_vocab(nlp)
        return Tokenizer(
            vocab,
            rules=rules,
            prefix_search=prefix_search,
            suffix_search=suffix_search,
            infix_finditer=infix_finditer,
            token_match=token_match,
        )


def self_to_spacy_tokens(func):
    """Decorates class method to first argument to a Fluent"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        wrapped_args = [SpacyTokens(args[0])] + list(args[1:]) if args else []
        return func(*wrapped_args, **kwargs)

    return wrapper


def regex(pattern):
    return lambda t: re.match(pattern, t.text) is not None


class SpacyTokens(Fluent):

    def __init__(self, iterable):
        if isinstance(iterable, str):
            iterable = (t for t in SpacyTokenizer.from_lang()(iterable))
            super(SpacyTokens, self).__init__(iterable)
        else:
            iterator = iter(self.create_iter(iterable))
            self._iterator = iterator

    def create_iter(self, iterable):
        for i in iterable:
            if isinstance(i, Token):
                yield i
            else:
                yield from (t for t in SpacyTokenizer.from_lang()(i))

    @staticmethod
    def to_token(string):
        global nlp
        if nlp is None:
            nlp = nlp_parser()
        return nlp(string)[0]

    @self_to_spacy_tokens
    def remove(self, func, *args, **kwargs):
        def _impl():
            for val in self._iterator:
                if not func(val, *args, **kwargs):
                    yield val

        return SpacyTokens(_impl())

    @self_to_spacy_tokens
    def remove_all(self, *funcs, **kwargs):
        def _impl():
            for val in self._iterator:
                if all(not func(val, **kwargs) for func in funcs):
                    yield val

        return SpacyTokens(_impl())

    @self_to_spacy_tokens
    def lemmatize(self):
        def _impl():
            yield from self.map(lambda t: SpacyTokens.to_token(t.lemma_))

        return SpacyTokens(_impl())

    @self_to_spacy_tokens
    def lower(self):
        def _impl():
            yield from self.map(lambda t: SpacyTokens.to_token(t.text.lower()))

        return SpacyTokens(_impl())


if __name__ == '__main__':
    # remove_all = SpacyTokens("It's good").remove_all(number, punct, regex("\d+"))
    # j = list(remove_all)
    corpus = ["test test", "test"]
    d = list(SpacyTokens(corpus).remove(punct))
    import pandas as pd

    df = pd.read_csv(
        "../notebooks/maria/train_dup.csv").drop_duplicates().dropna()
    corpus = pd.concat([df['question1'], df['question2']]).unique()
    d = {}
    for t in SpacyTokens(corpus).remove(punct):
        d[t.text.lower()] = d.get(t.text.lower(), 0) + 1
    utils.to_pickle(d, "../data/total_words.pkl")
