import itertools
from typing import Iterable

import spacy
import textacy
from nltk import FreqDist
from spacy.lang.en.stop_words import STOP_WORDS as stop_words
from spacy.lang.en import English
from string import punctuation as punctuations
from functools import lru_cache
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from spacy.language import Language
from spacy.parts_of_speech import NOUN, VERB
from spacy.tokens import Span
from spacy.util import compile_infix_regex
from textacy import Corpus, spacier
from textacy import vsm
from spacy_hunspell import spaCyHunSpell

import utils
from utils.misc import chunk

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


def process_question(question):
    en_doc = nlp_parser()(u'' + question)
    sent_list = list(en_doc.sents)
    sent = sent_list[0]
    wh_bi_gram = []
    root_token = ""
    wh_pos = ""
    wh_nbor_pos = ""
    wh_word = ""
    for token in sent:
        if token.tag_ == "WDT" or token.tag_ == "WP" or token.tag_ == "WP$" or token.tag_ == "WRB":
            wh_pos = token.tag_
            wh_word = token.text
            wh_bi_gram.append(token.text)
            wh_bi_gram.append(str(en_doc[token.i + 1]))
            wh_nbor_pos = en_doc[token.i + 1].tag_
        if token.dep_ == "ROOT":
            root_token = token.tag_

    return (wh_pos, wh_word, wh_bi_gram, wh_nbor_pos, root_token)


def subject_verb_object_triples(doc):
    """
    Extract an ordered sequence of subject-verb-object (SVO) triples from a
    spacy-parsed doc. Note that this only works for SVO languages.
    Args:
        doc (:class:`spacy.tokens.Doc` or :class:`spacy.tokens.Span`)
    Yields:
        Tuple[:class:`spacy.tokens.Span`]: The next 3-tuple of spans from ``doc``
        representing a (subject, verb, object) triple, in order of appearance.
    """
    # TODO: What to do about questions, where it may be VSO instead of SVO?
    # TODO: What about non-adjacent verb negations?
    # TODO: What about object (noun) negations?
    if isinstance(doc, Span):
        sents = [doc]
    else:  # spacy.Doc
        sents = doc.sents

    for sent in sents:
        start_i = sent[0].i

        verbs = spacier.utils.get_main_verbs_of_sent(sent)
        for verb in verbs:
            subjs = spacier.utils.get_subjects_of_verb(verb)
            if not subjs:
                continue
            objs = spacier.utils.get_objects_of_verb(verb)
            if not objs:
                continue

            # add adjacent auxiliaries to verbs, for context
            # and add compounds to compound nouns
            verb_span = spacier.utils.get_span_for_verb_auxiliaries(verb)
            verb = sent[verb_span[0] - start_i: verb_span[1] - start_i + 1]
            for subj in subjs:
                subj = sent[
                       spacier.utils.get_span_for_compound_noun(subj)[0]
                       - start_i: subj.i
                                  - start_i
                                  + 1
                       ]
                for obj in objs:
                    if obj.pos == NOUN:
                        span = spacier.utils.get_span_for_compound_noun(obj)
                    elif obj.pos == VERB:
                        span = spacier.utils.get_span_for_verb_auxiliaries(obj)
                    else:
                        span = (obj.i, obj.i)
                    obj = sent[span[0] - start_i: span[1] - start_i + 1]

                    yield (subj, verb, obj)


def create_words_dist(corpus, tokenizer=spacy_tokenizer):
    fdist = None
    for sentences in chunk(corpus, 5000):
        sentence = " ".join(sentences)
        if fdist is None:
            fdist = FreqDist(tokenizer(sentence))
        else:
            fdist.update(FreqDist(tokenizer(sentence)))
    return fdist


def word_counter(filename):
    f"wc -w {filename}"
    pass


def create_spell_dict(words):
    import platform
    sysname = platform.system().lower()
    nlp = nlp_parser()
    nlp.add_pipe(spaCyHunSpell(nlp, sysname))
    result = dict()
    for w in words:
        h = nlp(w)[0]
        if h._.hunspell_suggest and h._.hunspell_suggest[0] != w:
            result[w] = h._.hunspell_suggest[0]
    return result


if __name__ == '__main__':
    # "https://github.com/mpuig/spacy-lookup""
    # "https://github.com/manish-vi/Quora-Question-Similarity/blob/master/Quora_Question_Similarity_Case_Study.ipynb"
    # WH word | verb | subj | obj | ents
    df = pd.read_csv(
        "../notebooks/maria/train_dup.csv").drop_duplicates().dropna()
    corpus = pd.concat([df['question1'], df['question2']]).unique()
    # fdist = create_words_dist(corpus)
    d = utils.from_pickle("../data/words.pkl")  # create_words_dist(corpus)
    m = {w: c for w, c in d.items() if c<100}

    # utils.to_pickle(d, "./words.pkl")
    d = create_spell_dict(m.keys())
    utils.to_pickle(d, "../data/spell_words.pkl")
    nlp = nlp_parser()
    sentence = "How does the Surface-Pro himself 4 compare with iPad Pro?"


    # sentence = "I went to the this shop."
    # r = process_question(sentence)
    # subj_verb_obj  = subject_verb_object_triples(nlp(sentence))
    # text_ext = textacy.extract.subject_verb_object_triples(nlp(sentence))
    # tokens = [token for token in spacy_tokenizer(sentence)]
    # print(tokens)

    # v = TfidfVectorizeEx.from_corpus(corpus)
    # trainq1_trans = v.transform(df['question1'].values)
    # trainq2_trans = v.transform(df['question2'].values)

    #####
    def keep_token(t):
        return (t.is_alpha and
                not (t.is_space or t.is_punct or
                     t.is_stop or t.like_num))


    def lemmatize_doc(doc):
        return [t.lemma_ for t in doc if keep_token(t)]


    docs = [lemmatize_doc(nlp(doc)) for doc in list(df['question1'].values[:100])]
    from gensim.corpora import Dictionary
    from gensim.models.tfidfmodel import TfidfModel
    from gensim.matutils import sparse2full

    docs_dict = Dictionary(docs)
    # docs_dict.filter_extremes(no_below=20, no_above=0.2)
    # docs_dict.compactify()
    import numpy as np

    docs_corpus = [docs_dict.doc2bow(doc) for doc in docs]
    model_tfidf = TfidfModel(docs_corpus, id2word=docs_dict)
    docs_tfidf = model_tfidf[docs_corpus]
    docs_vecs = np.vstack([sparse2full(c, len(docs_dict)) for c in docs_tfidf])
    tfidf_emb_vecs = np.vstack([nlp(docs_dict[i]).vector for i in range(len(docs_dict))])
    docs_emb = np.dot(docs_vecs, tfidf_emb_vecs)
    from sklearn.decomposition import PCA

    docs_pca = PCA(n_components=8).fit_transform(docs_emb)
    from sklearn import manifold

    tsne = manifold.TSNE()
    viz = tsne.fit_transform(docs_pca)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.margins(0.05)

    zero_indices = np.where(df["is_duplicate"] == 0)[0]
    one_indices = np.where(df["is_duplicate"] == 1)[0]

    ax.plot(viz[zero_indices, 0], viz[zero_indices, 1], marker='o', linestyle='',
            ms=8, alpha=0.3, label="is not duplicated")
    ax.plot(viz[one_indices, 0], viz[one_indices, 1], marker='o', linestyle='',
            ms=8, alpha=0.3, label="is_duplicate")
    ax.legend()

    plt.show()
