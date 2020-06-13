import pandas as pd
import numpy as np
import fasttext
from flair.embeddings import WordEmbeddings, ELMoEmbeddings,\
    DocumentPoolEmbeddings, FastTextEmbeddings
from flair.data import Sentence
from scipy.spatial.distance import cosine
import pickle

class Embeder:
    def __init__(self):
        #https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M-subword.zip
        embedding = FastTextEmbeddings('../models/crawl-300d-2M-subword.bin')
        # embedding = WordEmbeddings('glove')
        self.document_embeddings = DocumentPoolEmbeddings([embedding])


    def embed_sent(self, sent):
        sentence = Sentence(sent)
        self.document_embeddings.embed(sentence)
        return sentence.get_embedding().cpu().detach().numpy()


def vectorize_file_pkl(input_file, output_file):
    import swifter
    embedder = Embeder()
    print("Vectorize started.")
    df = pickle.load(open('uni_enc.pkl', 'rb'))
    # df = df[:10]

    df["vectorize_lemma_q1"] = ""
    df['vectorize_lemma_q1'] = df['preprocessed_q1'].swifter.apply(embedder.embed_sent)
    print("Vectorize lemma q1 finished.")

    df["vectorize_lemma_q2"] = ""
    df['vectorize_lemma_q2'] = df['preprocessed_q2'].swifter.apply(embedder.embed_sent)
    print("Vectorize lemma q2 finished.")
    f = open(output_file, "wb")
    pickle.dump(df, f)

def cosine_dist(df, col_name):
    print("Start cosine distance")
    df[col_name] = 0
    df[col_name] = np.vectorize(cosine)(df['vectorize_lemma_q1'], df['vectorize_lemma_q2'])
    print("finished cosine distance for lemma vectors")
    return df


if __name__ == '__main__':
    vectorize_file_pkl('uni_enc.pkl', 'vectorize_lemma_all.pkl')
    f = open('vectorize_lemma_all.pkl', 'rb')
    df = pickle.load(f)
    df = cosine_dist(df, 'cos_dist_lemma')
    f.close()

    f = open("vectorize_lemma_all_cos.pkl", "wb")
    pickle.dump(df, f)

    # print(np.vectorize(cosine)(df['vectorize_q1'].to_numpy(), df['vectorize_q2'].to_numpy()))

    # df['cosine'] = df.apply(lambda row: 1 - cosine(row['vectorize_q1'], row['vectorize_q2']), axis=1)
    # print(df['cosine'])


    # embedder = Embeder()
    # vec = embedder.embed_sent("This is a test,")
    # print(vec)