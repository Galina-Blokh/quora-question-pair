import re
from functools import lru_cache, wraps

import spacy
from flupy.fluent import Fluent
from spacy.lang.en import English
from spacy.language import Language
from spacy.tokenizer import Tokenizer
from spacy.tokens import Doc, Token
import utils
from utils.pipeline import Pipeline
from nlp.stop_words import STOP_WORDS, QUESTION_WORDS


def stop_words(tok):
    if str(tok) in STOP_WORDS:
        return tok


space = lambda t: t.is_space


def punct(t):
    return t.is_punct


def stop(t):
    return t.is_stop


def question(t):
    return t.text.lower() in QUESTION_WORDS


def number(t):
    return t.like_num


def regex(pattern):
    return lambda t: re.match(pattern, t.text) is not None


nlp = None


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


def self_to_spacy_tokens(func):
    """Decorates class method to first argument to a Fluent"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        wrapped_args = [SpacyTokens(args[0])] + list(args[1:]) if args else []
        return func(*wrapped_args, **kwargs)

    return wrapper


class SpacyTokens(Fluent):

    def __init__(self, iterable):
        nlp = nlp_parser()
        tokenizer = nlp.tokenizer
        if isinstance(iterable, str):
            iterable = (t for t in tokenizer(iterable))
            super(SpacyTokens, self).__init__(iterable)
        else:
            iterator = (doc for doc in nlp.pipe(iterable, disable=["tagger", "parser"], n_threads=4, batch_size=10000))
            self._iterator = iterator

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

    @self_to_spacy_tokens
    def upper(self):
        def _impl():
            yield from self.map(lambda t: SpacyTokens.to_token(t.text.upper()))

        return SpacyTokens(_impl())

    @self_to_spacy_tokens
    def title(self):
        def _impl():
            yield from self.map(lambda t: SpacyTokens.to_token(t.text.title()))

        return SpacyTokens(_impl())


if __name__ == '__main__':
    pass
