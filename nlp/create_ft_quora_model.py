import pickle
import fasttext
import pandas as pd

def make_fasstext_model(df_name = 'preprocess_all_lev.csv',
                        text_file_name = 'corpus_lemmatized.txt', embedding_dim=128,
                        question1_column = 'preprocessed_q1', question2_column = 'preprocessed_q2',
                        save_or_not = True, name_for_saved_model = "model_corpus_lemmatized.bin"):
  '''
  Takes csv filename,
  text_file_name to save question corpus
  embedding dimension
  question columns names
  bool to save the model or not and file_name for saved model
  Returns fattext model
  '''

  df = pd.read_csv(df_name).drop(columns=['id', 'qid1', 'qid2', 'is_duplicate', 'Unnamed: 0'], axis=1)
  df = df.dropna()
  corpus_lemmatized = list(set(df[question1_column]).union(set(df[question2_column])))

  with open(text_file_name, "w", encoding="utf-8") as outfile:
      outfile.write("\n".join(str(item) for item in corpus_lemmatized))
  model_corpus_lemmatized = fasttext.train_unsupervised(text_file_name, model='skipgram', dim=embedding_dim)
  if save_or_not:
    model_corpus_lemmatized.save_model(name_for_saved_model)
  return model_corpus_lemmatized

m = make_fasstext_model()