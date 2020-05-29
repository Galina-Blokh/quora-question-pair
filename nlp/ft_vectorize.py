import pandas as pd
import numpy as np
import fasttext
from flair.embeddings import WordEmbeddings, ELMoEmbeddings,\
    DocumentPoolEmbeddings, FastTextEmbeddings
from flair.data import Sentence

class Embeder:
    def __init__(self):
        #https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M-subword.zip
        # embedding = FastTextEmbeddings('../models/crawl-300d-2M-subword.bin')
        embedding = WordEmbeddings('glove')
        self.document_embeddings = DocumentPoolEmbeddings([embedding])


    def embed_sent(self, sent):
        sentence = Sentence(sent)
        self.document_embeddings.embed(sentence)
        vector = sentence.get_embedding().cpu().detach().numpy()
        return vector


def vectorize_file_csv(input_file, output_file):
    import swifter
    embedder = Embeder()
    print("Vectorize started.")
    df = pd.read_csv(input_file).dropna()

    df["vectorize_q1"] = ""
    df['vectorize_q1'] = df['question1'].swifter.apply(embedder.embed_sent)
    print("Vectorize q1 finished.")

    df["vectorize_q2"] = ""
    df['vectorize_q2'] = df['question2'].swifter.apply(embedder.embed_sent)
    print("Vectorize q2 finished.")
    df.to_csv(output_file)



if __name__ == '__main__':
    vectorize_file_csv('preprocess_all_lev.csv', 'vectorize_all_lev.csv')

    # embedder = Embeder()
    # vec = embedder.embed_sent("This is a test,")
    # print(vec)