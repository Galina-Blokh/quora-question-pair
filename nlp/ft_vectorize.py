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


def vectorize_file_csv(input_file, output_file):
    import swifter
    embedder = Embeder()
    print("Vectorize started.")
    df = pd.read_csv(input_file).dropna()
    # df = df[:10]

    df["vectorize_q1"] = ""
    df['vectorize_q1'] = df['question1'].swifter.apply(embedder.embed_sent)
    print("Vectorize q1 finished.")

    df["vectorize_q2"] = ""
    df['vectorize_q2'] = df['question2'].swifter.apply(embedder.embed_sent)
    print("Vectorize q2 finished.")
    f = open(output_file, "wb")
    pickle.dump(df, f)

def cosine_dist(df, col_name):
    print("Start cosine distance")
    df[col_name] = 0
    df[col_name] = np.vectorize(cosine)(df['vectorize_q1'], df['vectorize_q2'])
    print("finished cosine distance")
    return df


if __name__ == '__main__':
    vectorize_file_csv('preprocess_all_lev.csv', 'vectorize_all_lev2.pkl')
    f = open('vectorize_all_lev2.pkl', 'rb')
    df = pickle.load(f)
    df = cosine_dist(df, 'cos_dist')
    f.close()

    f = open("vectorize_all_cos.pkl", "wb")
    pickle.dump(df, f)

    # print(np.vectorize(cosine)(df['vectorize_q1'].to_numpy(), df['vectorize_q2'].to_numpy()))

    # df['cosine'] = df.apply(lambda row: 1 - cosine(row['vectorize_q1'], row['vectorize_q2']), axis=1)
    # print(df['cosine'])


    # embedder = Embeder()
    # vec = embedder.embed_sent("This is a test,")
    # print(vec)