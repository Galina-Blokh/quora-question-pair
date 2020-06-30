import numpy
import pickle
from nearpy import Engine
from nearpy.hashes import RandomBinaryProjections
import train_models
import pandas as pd
from nearpy.distances import CosineDistance
from nearpy.filters import NearestFilter
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

class SimilarQuestionFinder:
    """
    Class to find similar questions in Quora Dataset
    single_feats_file - pickled df with for single Questions
    """
    def __init__(self, single_feats_file = 'models/single_feats.pkl',\
                        ft_model = 'models/model_corpus_lemmatized.bin',\
                        probs_model= 'models/rfc_for_probas.pkl',\
                        final_model = 'models/rfc_final.pkl',\
                        pair_model_file='models/rfc_for_probas.pkl'):
        self.df = pickle.load(open(single_feats_file, 'rb'))
        self.sfe = train_models.SingleFeatsExctractor(ft_model)
        self.pfe = train_models.PairFeatsExctractor(pair_model_file)
        self.rfc_probs = pickle.load(open(probs_model, 'rb'))
        self.rfc_final = pickle.load(open(final_model , 'rb'))


        # Dimension of our vector space
        dimension = len(self.df['ft_vector_q1'][0])
        print('dimension: ', dimension)

        # Create a random binary hash with 10 bits
        rbpt = RandomBinaryProjections('rbp1', 10)

        # Create engine with pipeline configuration
        self.engine = Engine(dimension, lshashes=[rbpt], distance=CosineDistance(),\
                        vector_filters=[NearestFilter(30)])

        # Index 1000000 random vectors (set their data to a unique string)
        for index in range(len(self.df)):
            v = self.df['ft_vector_lem_q1'][index]
            self.engine.store_vector(v, index)


    def prepare_question(self, question):
        """
        prepare feats for inputed question
        :param question: srting
        :return: df with nearest (nearpy) questions from Quora and feats
        """

        question_feats = self.sfe.exctract_feats_from_text(question)
        # Get nearest neighbours
        N = self.engine.neighbours(question_feats['ft_vector_lem'])

        neighbour_list = []

        if len(N) ==0:
            return None
        for neighbour in N:
            neighbour_list.append(self.df.iloc[neighbour[1], :])

        df_question = pd.DataFrame.from_dict([question_feats] * len(neighbour_list))

        df_neighbour = pd.DataFrame(neighbour_list).reset_index()

        df_question = df_question.add_suffix('_q2')

        df_final = df_neighbour.drop(['lem_str_q2', 'ner_q2',
                           'ft_vector_q2', 'ft_vector_lem_q2', 'question2_q2'], axis=1)
        df_final['ft_vector_q2'] = df_question['ft_vector_q2']
        df_final['ft_vector_lem_q2'] = df_question['ft_vector_lem_q2']
        df_final['ner_q2'] = df_question['ner_q2']
        df_final['lem_str_q2'] = df_question['lem_str_q2']

        df_final = self.pfe.extract_pair_feats(df_final)

        X_train = np.concatenate((np.stack(df_final['ft_vector_lem_q1'].to_numpy(), axis=0), \
                                  np.stack(df_final['ft_vector_lem_q2'].to_numpy(), axis=0)), axis=1)


        rfc_probas_train = self.rfc_probs.predict_proba(X_train)
        rfc_probas_train_0 = [x[0] for x in rfc_probas_train]
        df_final['rfc_probas_0'] = rfc_probas_train_0
        return df_final


    def get_similars_for_question(self, question, threshold=0.5):
        """
        Get similar questions for th input question
        :param question: string
        :param threshold: threshold for predict probas
        :return: df with similar questions and predict probas or empty df if there is no similar questions
        """
        df_final = self.prepare_question(question)
        if df_final is None:
            print("We don't have similar questions")
            return pd.DataFrame()
        pred_final = self.rfc_final.predict_proba(df_final[train_models.FEATS])
        df_final['pred'] = [x[1] for x in pred_final]
        df_return = df_final[['question1_q1', 'pred']]

        return df_return[df_return['pred'] > threshold].sort_values(by='pred', ascending=False)



if __name__ == '__main__':

    # print(df.head())
    # print(df.columns)
    # engine = prepare_search_engine(self.df)
    sim_qf = SimilarQuestionFinder()

    questions = ['Did Trump win the election?', 'How to learn chineeze?',\
               'What is the best town?', 'What is the funniest joke?',\
               'What is the best way to learn English?']
    for question in questions:
        print(question)
        print(sim_qf.get_similars_for_question(question))
