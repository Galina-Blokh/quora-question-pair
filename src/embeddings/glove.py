import os

import numpy as np
from typing import List

from tqdm import tqdm
from definitions import PROJECT_DIR
from utils import download_file
import subprocess
import pandas as pd

class Glove2Vec():
    def __init__(self):
        self.model: dict = dict()

    def load_model(self, filename: str):
        self.model = {}
        f = open(filename)
        for line in tqdm(f):
            values = line.split()
            word = values[0]
            try:
                coefs = np.asarray(values[1:], dtype='float32')
                self.model[word] = coefs
            except:
                pass
        f.close()
        return self

    def embed_sentence(self, in_object: str) -> np.array:
        if in_object in self.model:
            return self.model[in_object]
        return np.zeros((300,))

    def embed_sentences(self, in_objects: List) -> List:
        return [
            self.embed_sentence(in_object)
            for in_object in tqdm(in_objects, total=len(in_objects))
        ]


def create_embeddings(in_file: str, out_file: str):
    in_data = pd.read_csv(in_file)
    assert "question1" in in_data.columns
    assert "question2" in in_data.columns
    in_data = in_data[(~in_data["question1"].isna()) & (~in_data["question2"].isna())]
    in_data["question1"] = in_data["question1"].str.lower()
    in_data["question2"] = in_data["question2"].str.lower()

    model_file = os.path.join(PROJECT_DIR, "data", "external", "glove.840B.300d.txt")
    if not os.path.exists(model_file):
        print("downloading model file...")
        from pathlib import Path
        model_file_zip = Path(model_file).with_suffix('.zip')
        download_file(url="http://www-nlp.stanford.edu/data/glove.840B.300d.zip", filename=model_file_zip)
        subprocess.run(["unzip", model_file_zip, "-d", os.path.dirname(model_file)])
        subprocess.run(["rm", model_file_zip])
    model = Glove2Vec()
    model.load_model(model_file)



"""
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk import punkt
stop_words = stopwords.words('english')

class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

train = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/train.csv').fillna(' ')
test = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/test.csv').fillna(' ')

train_text = train['comment_text']
test_text = test['comment_text']
all_text = pd.concat([train_text, test_text])

print("Checkpoint1 - Data Read Complete")

embeddings_index = {}
f = open('../input/glove840b300dtxt/glove.840B.300d.txt', encoding="utf8")
for line in tqdm(f):
    values = line.split()
    word = values[0]
    try:
       coefs = np.asarray(values[1:], dtype='float32')
       embeddings_index[word] = coefs
    except ValueError:
       pass
f.close()
print('Found %s word vectors.' % len(embeddings_index))
# this function creates a normalized vector for the whole sentence
def sent2vec(s):
    words = str(s).lower()
    words = word_tokenize(words)
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(embeddings_index[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    if type(v) != np.ndarray:
        return np.zeros(300)
    return v / np.sqrt((v ** 2).sum())

# create sentence vectors using the above function for training and validation set
xtrain_glove = [sent2vec(x) for x in tqdm(train_text)]
xtest_glove = [sent2vec(x) for x in tqdm(test_text)]

print('Checkpoint2 -Normalized Vector for Sentences are created')

xtrain_glove = np.array(xtrain_glove)
xtest_glove = np.array(xtest_glove)

scores = []
submission = pd.DataFrame.from_dict({'id': test['id']})
for class_name in class_names:
    train_target = train[class_name]
    classifier = LogisticRegression(solver='sag')

    cv_score = np.mean(cross_val_score(classifier, xtrain_glove, train_target, cv=3, scoring='roc_auc'))
    scores.append(cv_score)
    print('CV score for class {} is {}'.format(class_name, cv_score))

    classifier.fit(xtrain_glove, train_target)
    print('Training LogisticRegression Classifier for {} is complete!!'.format(class_name))
    submission[class_name] = classifier.predict_proba(xtest_glove)[:, 1]

print('Total CV score is {}'.format(np.mean(scores)))

submission.to_csv('submission_glove_LogisticRegression.csv', index=False)
"""


if __name__ == '__main__':
    import fire

    fire.Fire()
