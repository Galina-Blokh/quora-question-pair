import spacy
import sys
sys.path.append("..")
from nlp import preprocess_data
import logging
import argparse
import pandas as pd
import numpy as np
from nlp import ner
from nlp import ft_vectorize
from nlp import wer_lev
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from scipy.spatial.distance import cosine

LOG_FILE = 'train_model.log'

# Pipline:
# Tokenization + lematization ( preprocess_data.preprocess_file_csv3)
# Levinshtain distance (wer_lev) (on lemmatized questions)
# NER (ner.ner_df_file_pkl) (on non-lemmatizeed questions)
# FastText vectors non lematized (ft_vectorize.vectorize_file_csv)
# FastText  non lematized cos distance (ft_vectorize.cosine_dist)
# FastText lematized vecs+cosin dist  (ft_vectorize_lemma.cosine_dist vectorize_file_pkl)
# Encoder-decoder score (uni_encoder.py)

# Train your model (rfc_final.py - RFC)

class SingleFeatsExctractor:
    def __init__(self, ft_model_file):
        self.ner_extractor = ner.NER()
        self.ft_vectorizer = ft_vectorize.Embeder(ft_model_file)

    def exctract_feats_from_text(self, str):
        """
        tokenization, lemmatization, FastText vectors, NER (without NER score)
        :param str: text
        :return: features
        """
        feats = {}
        feats['lem_str'] = preprocess_data.preprocess_sent3(str)
        feats['ner'] = self.ner_extractor.recognize(str)
        feats['ft_vector'] = self.ft_vectorizer.embed_sent(str)
        feats['ft_vector_lem'] = self.ft_vectorizer.embed_sent(feats['lem_str'])
        return feats

def extract_single_train_feats(input_file, ft_model, out_pikle='single_feats.pkl'):
    extractor = SingleFeatsExctractor(ft_model)
    logger.info("Load data.")
    df = pd.read_csv(input_file)
    df=df[:1000]
    logger.info("Extract feats for Q1.")
    ser_feats_q1 = df['question1'].apply(extractor.exctract_feats_from_text)
    df_feats_q1 = pd.DataFrame(ser_feats_q1.tolist())
    df_feats_q1['question1'] = df['question1']
    df_feats_q1 = df_feats_q1.add_suffix('_q1')
    df_feats_q1.rename(index=str, columns={'headertable1': 'header'})

    logger.info("Extract feats for Q2.")
    ser_feats_q2 = df['question2'].apply(extractor.exctract_feats_from_text)
    df_feats_q2 = pd.DataFrame(ser_feats_q2.tolist())
    df_feats_q2['question2'] = df['question2']
    df_feats_q2 = df_feats_q2.add_suffix('_q2')
    df_feats_q2.rename(index=str, columns={'headertable1': 'header'})
    df_feats = pd.concat([df_feats_q1, df_feats_q2], axis=1, sort=False)
    df_feats['is_duplicate'] = df['is_duplicate']

    logger.info("Extract single features: done")
    pickle.dump(df_feats, open(out_pikle, 'wb'))
    logger.info('Single feats saved to single_feats.pkl')


def train_on_vecs(pkl_file='single_feats.pkl', out_file ='rfc_for_probas.pkl'):

    df = pickle.load(open(pkl_file, 'rb'))
    df_train, df_test = train_test_split(df, test_size=0.10,random_state=13)

    ### concatenate q1 and q2 fast_text vectors
    X_train = np.concatenate((np.stack(df_train['ft_vector_lem_q1'].to_numpy(), axis=0),\
                              np.stack(df_train['ft_vector_lem_q2'].to_numpy(), axis=0)), axis=1)
    X_test = np.concatenate((np.stack(df_test['ft_vector_lem_q1'].to_numpy(), axis=0),\
                              np.stack(df_test['ft_vector_lem_q2'].to_numpy(), axis=0)), axis=1)

    ### Target features
    y_test = df_test['is_duplicate']
    y_train = df_train['is_duplicate']

    ### train model only on fasttext vectors
    logger.info("Start training RFC for ft-vectors")
    rfc = RandomForestClassifier(n_estimators = 600,n_jobs = 8)
    rfc.fit(X_train, y_train)
    y_pred = rfc.predict(X_test)
    logger.info('\n clasification report (RFC (200 est) + fast_text_vectors:\n' +\
           str(classification_report(y_test, y_pred)))

    pickle.dump(rfc, open(out_file, 'wb'))
    logger.info('Saved vectors rfc to' + out_file)


class PairFeatsExctractor:
    def __init__(self, pair_model_file='rfc_for_probas.pkl'):
        self.ner_extractor = ner.NER()
        self.rfc = pickle.load(open(pair_model_file, 'rb'))


    def train_final_model(self,df_train, df_test,out_model_file='rfc_final.pkl'):
        X_train = np.concatenate((np.stack(df_train['ft_vector_lem_q1'].to_numpy(), axis=0), \
                                  np.stack(df_train['ft_vector_lem_q2'].to_numpy(), axis=0)), axis=1)
        X_test = np.concatenate((np.stack(df_test['ft_vector_lem_q1'].to_numpy(), axis=0), \
                                 np.stack(df_test['ft_vector_lem_q2'].to_numpy(), axis=0)), axis=1)

        y_train = df_train['is_duplicate']
        y_test = df_test['is_duplicate']

        rfc_probas_train = self.rfc.predict_proba(X_train)
        rfc_probas_test = self.rfc.predict_proba(X_test)

        rfc_probas_train_0 = [x[0] for x in rfc_probas_train]
        rfc_probas_test_0 = [x[0] for x in rfc_probas_test]

        df_train['rfc_probas_0'] = rfc_probas_train_0
        df_test['rfc_probas_0'] = rfc_probas_test_0

        feats = ['ner_score', 'wer', 'cos_dist', 'cos_dist_lem', 'rfc_probas_0']
        logger.info("Train columns")
        logger.info(str(df_train.columns))
        logger.info("Train fetas:")
        logger.info(str(feats))
        # print(df.info())

        # feats=['score']

        rfc_final = RandomForestClassifier(n_estimators=100, class_weight={0: 2, 1: 10}, \
                                           min_samples_split=8, min_samples_leaf=2)
        rfc_final.fit(df_train[feats], y_train)
        y_pred_final = rfc_final.predict(df_test[feats])

        logger.info('\n clasification report (RFC cw 2,10 + cosine distance + WER + predict_probas + NER:\n' \
                    + str(classification_report(y_test, y_pred_final)))
        pickle.dump(rfc_final, open(out_model_file, 'wb'))
        logger.info('rfc_final saved to rfc_final.pkl')

    def extract_pair_train_feats(self, input_pkl):
        logger.info("Extracting pair feats.")
        df = pickle.load(open(input_pkl, 'rb'))
        df_train, df_test = train_test_split(df, test_size=0.10, random_state=13)
        df_train = self.extract_pair_feats(df_train)
        df_test = self.extract_pair_feats(df_test)
        self.train_final_model(df_train,df_test)

    def extract_pair_feats(self, df):
        logger.info("Loaded dataframe from pikle")
        df['cos_dist'] = np.vectorize(cosine)(df['ft_vector_q1'], df['ft_vector_q2'])
        logger.info("Calculated cosine distance for non-lemmatized ft vectors")
        df['cos_dist_lem'] = np.vectorize(cosine)(df['ft_vector_lem_q1'], df['ft_vector_lem_q2'])
        logger.info("Calculated cosine distance for lemmatized ft vectors")
        df['wer'] = np.vectorize(wer_lev.lev_dist)(df['lem_str_q1'], df['lem_str_q2'])
        logger.info("Calculated WER for lemmatized ft vectors")
        df['ner_score'] = np.vectorize(self.ner_extractor.ner_score)(df['ner_q1'], \
                                                             df['ner_q2'])
        return df






def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    return logger




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train_model for question pairs')

    parser.add_argument('--train_csv', type=str, help='File with question pairs for training', action="store",
                        default='train_dup.csv')
    parser.add_argument('--embed_model', type=str, help='Fasttext unsupervised model',
                        default='models/small_en.bin')
    parser.add_argument('--rfc_ft_vectors', type=str, help='Where to store model ft_vectors model',
                        default='rfc_for_probas.pkl')
    parser.add_argument('--single_feats', type=str, help='Where to store df with single feats',
                        default='single_feats.pkl')
    # parser.add_argument('--competitions_api_file', type=str, help='Where to store data for competitions from API',
    #                     default='competitions_api_file.csv')
    # parser.add_argument('--kernels_file', type=str, help='Where to store kernels from competition page',
    #                     default='kernels_file.csv')
    args = parser.parse_args()

    logger = get_logger(__name__)
    logger.info("Prepare NER.")

    extract_single_train_feats(args.train_csv, args.embed_model, args.single_feats)
    train_on_vecs(args.single_feats, args.rfc_ft_vectors)
    pfe = PairFeatsExctractor()
    pfe.extract_pair_train_feats(args.single_feats)

