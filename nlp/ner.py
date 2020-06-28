import spacy
import pandas as pd
import numpy as np
import pickle

class NER:
    """
    Class NER-recognition (spacy), calculating NER-score: (equal NERs - different NERs)/ all NERS.
    equal - set of NERs are in q1 and q2, different - set of NERs are only in q1 or on;y in q2
    """
    def __init__(self, model = "en_core_web_md"):
        if model is not None:
            self.nlp = spacy.load(model)  # load existing spacy model
            print("Loaded model '%s'" % model)
        else:
            self.nlp = spacy.blank('en')  # create blank Language class
            print("Created blank 'en' model")

        if 'ner' not in self.nlp.pipe_names:
            self.ner = self.nlp.create_pipe('ner')
            self.nlp.add_pipe(self.ner)
        else:
            self.ner = self.nlp.get_pipe('ner')

    def recognize(self, text):
        """
        NER recognition (spacy for non-lemmatized text)
        :param text: text
        :return: NERs (without tags)
        """
        doc = self.nlp(text)
        result = []
        for ent in doc.ents:
            result.append(ent.text)
            #result.append(ent.label_ + ' ' + ent.text)    <-- return both NERs and NER-tags
        return "|".join(result).lower()

    def ner_score(self,str_1, str_2):
        """
        calculating NER-score: (equal NERs - different NERs)/ all NERS.
    equal - set of NERs are in q1 and q2, different - set of NERs are only in q1 or on;y in q2
        :param str_1: question_1 NERs
        :param str_2: question_2 NERs
        :return: score form -1 to 1
        """
        ners_1 = set(str_1.split("|"))
        ners_2 = set(str_2.split("|"))
        diff_ner = (ners_1 - ners_2).union(ners_2 - ners_1)
        all_ner = ners_1.union(ners_2)
        equal_ner = all_ner - diff_ner
        return (len(equal_ner) - len(diff_ner)) / len(all_ner)

def ner_df_file_pkl(df, output_pkl):
    """
    Applied NER recognition and claculating NER scores and save df to pickle file
    :param df: pandas df
    :param output_pkl: name of output pickle file
    :return: pickle file
    """

    import swifter
    recognizer = NER()
    print("NER started.")

    # df = df[:10]

    df["NER_q1"] = ""
    df['NER_q1'] = df['question1'].swifter.apply(recognizer.recognize)
    print("NER q1 finished.")

    df["NER_q2"] = ""
    df['NER_q2'] = df['question2'].swifter.apply(recognizer.recognize)
    print("NER q2 finished.")

    print("NER score calculation start")
    df['ner_score'] = np.vectorize(recognizer.ner_score)(df['NER_q1'], \
                                                         df['NER_q2'])
    print("NER score calculation finished")

    f = open(output_pkl, "wb")
    pickle.dump(df, f)




if __name__ == '__main__':
    df = pickle.load(open('vectorize_all_cos.pkl', 'rb'))
    ner_df_file_pkl(df, 'ner.pkl')

    #### for debug
    # df = pickle.load(open('test_ner_text.pkl', 'rb'))
    # df_debug = df[['question1', 'question2', "NER_q1", "NER_q2"]]
    # print(df_debug)