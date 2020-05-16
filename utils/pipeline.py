import subprocess
from itertools import tee, chain

from flupy import flu
from flupy.fluent import self_to_flu, Fluent
from nltk import ngrams

class Pipeline:
    pass
Pipeline = flu

@self_to_flu
def zip_with_index(self):
    def _impl():
        return enumerate(self)

    return Fluent(_impl())


@self_to_flu
def flat_map(self, func):
    def _impl():
        return self.flatten().map(func)

    return Fluent(_impl())


@self_to_flu
def prev_curr(self, fill_value=None):
    def _impl():
        prev, curr = tee(self, 2)
        prev = chain([fill_value], prev)
        return zip(prev, curr)

    return Fluent(_impl())

@self_to_flu
def split(self, splitter):
    def _impl():
        return splitter(self)

    return Fluent(_impl())

@self_to_flu
def remove(self, func, *args, **kwargs):
    def _impl():
        for val in self._iterator:
            if not func(val, *args, **kwargs):
                yield val
    return Fluent(_impl())

@self_to_flu
def remove_all(self, *funcs, **kwargs):
    def _impl():
        for val in self._iterator:
            if all(not func(val, **kwargs) for func in funcs):
                yield val
    return Fluent(_impl())

def by_ngrams(sequence, size=4):
    yield from Pipeline(ngrams(sequence, size)).map(lambda it: "".join(it))

def kmers(sequence):
    return by_ngrams(sequence, 6)

Pipeline.zip_with_index = zip_with_index
Pipeline.prev_curr = prev_curr
Pipeline.flat_map = flat_map
Pipeline.split = split
Pipeline.remove = remove
Pipeline.remove_all = remove_all