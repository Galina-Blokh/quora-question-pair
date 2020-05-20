#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import metrics
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
import spacy
from spacy.lang.en import English
spacy.load('en')
nlp = English()
import nlp
import string
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from string import punctuation as PUNCTS
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


# In[2]:


# CONSTANTS
PUNCTS = set(punct for punct in string.punctuation)
STOP_WORDS = set(stopwords.words("english")) - {'who', 'whom', 'what', 'when', 'where', 'why', 'how'}


# In[3]:


df = pd.read_csv('train_dup.csv').drop(columns='id', axis=1)
df = df.dropna()
df.head()


# In[4]:


duplicates = df[df.is_duplicate==1]
not_duplicates = df[df.is_duplicate == 0]
print(f'The proportion of classes pairs/not pairs is: {len(duplicates)}/{len(not_duplicates)}')
print(f'An example of duplicated questions: \n{duplicates.question1[5]} \nAND \n{duplicates.question2[5]}')
duplicates.head()


# In[5]:


# This is an example, will be later removed
count_vectorizer = CountVectorizer()
print(duplicates.question1[5], duplicates.question2[5])
# Create the Bag-of-Words Model
bag_of_words = count_vectorizer.fit_transform([duplicates.question1[5], duplicates.question2[5]])

# Show the Bag-of-Words Model as a pandas DataFrame
feature_names = count_vectorizer.get_feature_names()
pd.DataFrame(bag_of_words.toarray(), columns = feature_names)


# In[45]:


# nltk cleaning: very slow!
lemmatizer = WordNetLemmatizer()
def clean_sentence_lemmatizer(sentence):
    """
    Receives a raw sentence and clean it using the following steps:  # BETTER
    1. Remove all non-words
    2. Transform the review in lower case
    3. Remove all stop words
    4. Perform lemmatizer

    Args:
        sentence: the sentence that will be cleaned
    Returns:
        a clean sentence using the mentioned steps above.
    """
    
    sentence = re.sub("[^A-Za-z]", " ", sentence)
    sentence = sentence.lower()
    sentence = word_tokenize(sentence)
    sentence = [lemmatizer.lemmatize(word) for word in sentence if word not in STOP_WORDS]
    sentence = " ".join(sentence)
    return sentence


clean_sentence_lemmatizer('day days date dates shotgun shotguns wolf wolves child children')


# ## SPACY

# Tokenization, lemmatization, removing stop words (except question opening words like who, why, etc) and puctuation and lowering

# In[4]:


# 1st way to clean q = re.sub("[^A-Za-z]", " ", q)?
nlp = English()

clean_question = lambda sentence: ' '.join([word.lemma_.lower() for word in nlp(sentence) 
                                            if word.lemma_ not in STOP_WORDS if word.lemma_ not in PUNCTS])
df_clean = df.copy()
df_clean['question1'] = df.apply(lambda row: clean_question(row['question1']), axis=1)
df_clean['question2'] = df.apply(lambda row: clean_question(row['question2']), axis=1)
df_clean.head()


# In[5]:


resub = lambda q: re.sub("[^A-Za-z]", " ", q)
df_clean['question1'] = df_clean.apply(lambda row: resub(row['question1']), axis=1)
df_clean['question2'] = df_clean.apply(lambda row: resub(row['question2']), axis=1)


# In[6]:


df_clean.to_csv('df_clean.csv', index=False)


# In[10]:


# 2nd way to clean
# I didn't use it
def tok_stop_lem_punct(q, tokenize=True, stopwordize=True, punctuanize=True, lemmatize=True, lowerize=True):
    nlp = English()
    if stopwordize:
        q = ' '.join([words.text for words in nlp(q) if words.text not in STOP_WORDS])
    if lemmatize:
        q = ' '.join([words.lemma_ for words in nlp(q)])
    if punctuanize:
        q = re.sub("[^A-Za-z]", " ", q)
        q = ' '.join([words.strip() for words in q.split() if words not in PUNCTS])
    if lowerize:
        q = ' '.join([words.lower() for words in q.split()])
    if tokenize:
        tokenizer = spacy.tokenizer.Tokenizer(nlp.vocab)
        q = tokenizer(q)
    return q

print(tok_stop_lem_punct(duplicates.question1[5]))


# In[ ]:


df_clean2 = df.copy()
df_clean2['question1'] = df.apply(lambda row: tok_stop_lem_punct(row['question1']), axis=1)
df_clean2['question2'] = df.apply(lambda row: tok_stop_lem_punct(row['question2']), axis=1)


# Some basic stats

# In[136]:


unigramm_count_dict_clean, unigramm_count_dict_initial = {}, {}
bigramm_count_dict_initial, bigramm_count_dict_clean = {}, {}
number_of_words_inq_initial, number_of_words_inq_clean = [], []
lengths_of_questions_initial, lengths_of_questions_clean = [], []


def count_basic_stats(df):
    unigramm_count_dict = {}
    bigramm_count_dict = {} 
    number_of_words_inq = []
    lengths_of_questions = []
    
    for q1, q2 in zip(df.question1, df.question2):
        unigramms_q1 = [word.text for word in nlp(q1)]
        unigramms_q2 = [word.text for word in nlp(q2)]
        number_of_words_inq.append(len(unigramms_q1))
        number_of_words_inq.append(len(unigramms_q2))
        lengths_of_questions.append(len(q1))
        lengths_of_questions.append(len(q2))

        for unigramm in unigramms_q1 + unigramms_q2:
            if unigramm in unigramm_count_dict:
                unigramm_count_dict[unigramm] += 1
            else:
                unigramm_count_dict[unigramm] = 1

        bigramms_q1 = [' '.join(q1.split()[i:i+2]) for i in range(len(q1.split())) if i < len(q1.split()) - 1]
        bigramms_q2 = [' '.join(q2.split()[i:i+2]) for i in range(len(q2.split())) if i < len(q2.split()) - 1]

        for bigramm in bigramms_q1 + bigramms_q2:
            if bigramm in bigramm_count_dict:
                bigramm_count_dict[bigramm] += 1
            else:
                bigramm_count_dict[bigramm] = 1
    return unigramm_count_dict, bigramm_count_dict, number_of_words_inq, lengths_of_questions


unigramm_count_dict_initial, bigramm_count_dict_initial, number_of_words_inq_initial,     lengths_of_questions_initial = count_basic_stats(df)

unigramm_count_dict_clean, bigramm_count_dict_clean, number_of_words_inq_clean,     lengths_of_questions_clean = count_basic_stats(df_clean)


# In[153]:


# TF(term) = term_frequency / sum(all terms frequences)

tf_dict_initial = {unigramm : unigramm_count_dict_initial[unigramm] / sum(unigramm_count_dict_initial.values()) 
                  for unigramm in unigramm_count_dict_initial}
tf_dict_clean = {unigramm : unigramm_count_dict_clean[unigramm] / sum(unigramm_count_dict_clean.values())
                for unigramm in unigramm_count_dict_clean}


# In[68]:


from collections import Counter

def word_usage_count(df, column1, column2, ignore_register = False):
    """
    Receives a df with raw sentences and counts word usage
    Requires collections.Counter  
    Args:
        df: dataframe with some questions
        column1, column2: columns to unite and count word usage
        ignore_register: boolean parameter to ignore register or not, defaults to False
    Returns:
        Counter object {word: number}.
    """
    questions1 = set(df[column1])
    questions2 = set(df[column2])
    unique_question_union = questions1.union(questions2)
    words_usage_count = Counter()
    for q in unique_question_union:
        for word in q.split():
            if ignore_register:
                words_usage_count[word.lower()]+= 1
            else:
                words_usage_count[word]+= 1
    return words_usage_count

def word_existence_in_corpus(df,column1, column2, ignore_register=False):
    """
    Receives a df with raw sentences and counts word usage
    Requires collections.Counter  
    Args:
        df: dataframe with some questions
        column1, column2: columns to unite and get words
        ignore_register: boolean parameter to ignore register or not, defaults to False
    Returns:
        set of words occured in corpus from df.
    """
    return set(word_usage_count(df, column1, column2, ignore_register = ignore_register).elements())


import time
t0 = time.time()
total_word_dict = word_usage_count(df,'question1', 'question2',)
t1 = time.time()
print(len(total_word_dict))
print(t1 - t0)
print(len(word_existence_in_corpus(df,'question1', 'question2',)))


# In[ ]:





# Playing with vectorizer
# 
# took processed data
# 

# In[3]:


df_p = pd.read_csv('preprocess_all.csv')
df_p = df_p.drop(columns=['id', 'Unnamed: 0'])
df_p.head()


# In[4]:


df_clean = pd.read_csv('df_clean.csv')
df_clean.head()


# ## 1. Vectorization from full corpus

# In[ ]:





# In[65]:


# my clean
df_clean = pd.read_csv('df_clean.csv')
df_clean = df_clean.dropna()
print(f'Initial cleaned data shape {df_clean.shape}')
from sklearn.feature_extraction.text import TfidfVectorizer
corpus_clean = set(df_clean.question2).union(set(df_clean.question1))
vectorizer_clean = TfidfVectorizer()
X_clean = vectorizer_clean.fit_transform(corpus_clean)
print(f'TfIdfVectorizer shape {X_clean.shape}')

from sklearn.feature_extraction.text import CountVectorizer
cv_clean = CountVectorizer()
X2_clean = cv_clean.fit_transform(corpus_clean)
print(f'Count Vectorizer/bag of words shape {X2_clean.shape}')


# In[41]:


# this is good df
df_p = df_p.dropna()
print(f'Initial preprocessed data shape {df_p.shape}')

corpus_p = set(df_p.preprocessed_q1).union(set(df_p.preprocessed_q1))
vectorizer_p = TfidfVectorizer()
X_p = vectorizer_p.fit_transform(corpus_p)
print(f'TfIdfVectorizer shape {X_p.shape}')

cv_p = CountVectorizer(max_features=1000)
X2_p = cv_p.fit_transform(corpus_p)
print(f'Count Vectorizer/bag of words shape with max_features 1000 {X2_p.shape}')


# ## 2. Transform first 10000 rows with CountVectorizer

# In[ ]:


# nevermind))
# t0 = time.time()
# c_vectorized_q1 = cv_clean.transform(df_clean.head(10000).question1)
# c_vectorized_q2 = cv_clean.transform(df_clean.head(10000).question2)
# print(f'Count Vectorizer transformed first 10000 rows into sparse matrix of shape {df_p_vectorized_q1.shape}')
# df_p_vectorized_cv_q1 = pd.DataFrame(df_p_vectorized_q1.toarray(), columns = cv_clean.get_feature_names())
# df_p_vectorized_cv_q2 = pd.DataFrame(df_p_vectorized_q2.toarray(), columns = cv_clean.get_feature_names())
# print(f'We can applyt todense() and take df from that, that will be again shape {df_p_vectorized_cv_q1.shape}')
# print('First few rows from that df: \n')
# df_p_vectorized_cv_q1.head(2)


# In[66]:


# let's try Count Vectorizer on first 10000 rows - transform q1 and q2 to vectors
import time
t0 = time.time()
df_p_vectorized_q1 = cv_p.transform(df_p.head(10000).preprocessed_q1)
df_p_vectorized_q2 = cv_p.transform(df_p.head(10000).preprocessed_q2)
print(f'Count Vectorizer transformed first 10000 rows into sparse matrix of shape {df_p_vectorized_q1.shape}')
df_p_vectorized_cv_q1 = pd.DataFrame(df_p_vectorized_q1.toarray(), columns = cv_p.get_feature_names())
df_p_vectorized_cv_q2 = pd.DataFrame(df_p_vectorized_q2.toarray(), columns = cv_p.get_feature_names())
print(f'We can applyt todense() and take df from that, that will be again shape {df_p_vectorized_cv_q1.shape}')
print('First few rows from that df: \n')
df_p_vectorized_cv_q1.head(2)


# ## 3. Finding cosine distance between q1[i] from question1 and q2[i] from question2 
# diagonal of pairwise distances between to matrices

# In[75]:


# counting cosine similarity between first 10000 qs
from sklearn.metrics.pairwise import cosine_similarity # aka 1 - cosine distance
from scipy.spatial.distance import cosine


def count_cosine_dist_between_two_dfs(sparse_m1, sparse_m2):
    """
    Receives  sparces matrices and counts pairwise cosine distancer  
    Args:
        matrices
        
    Returns:
        list of pairwise distances
    """
    
    return [1 - cosine_similarity(sparse_m1.todense()[i, :], sparse_m2.todense()[i, :])[0][0] 
            for i in range((min(sparse_m1.shape[0], sparse_m2.shape[0])))]

distancesq1iq2i = count_cosine_dist_between_two_dfs(df_p_vectorized_q1, df_p_vectorized_q2)
print(f'Array of distances between q1_i and q2_i has length {len(distancesq1iq2i)} - should be {len(df_p_vectorized_cv_q1)}')
print(f'First 10 questions have cosine distance: \n{distancesq1iq2i[:10]}')


# ## 4. Creating DF with cosine distance and target

# In[77]:


#distance_target = pd.DataFrame([distancesq1iq2i, df_clean.is_duplicate[:10000].values], columns= ['dist', 'target'])
distance_target = pd.DataFrame({'distance': distancesq1iq2i, 'target': df_clean.is_duplicate[:10000].values})
distance_target.describe()


# In[79]:


X = distance_target.distance
y = distance_target.target
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)
print('data shapes', X_train.shape, X_test.shape, y_train.shape)
print(f'Distribution of classes in target: \n{y_train.value_counts()/len(y_train)*100}, \n{y_test.value_counts()/len(y_test)*100}')
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train.values.reshape(-1, 1), y_train)
y_pred = lr.predict(X_test.values.reshape(-1, 1))

from sklearn.metrics import confusion_matrix, classification_report
print(f'Classification report after LR: \n{classification_report(y_test, y_pred)}')
print(f'Confusion matrix after LR: \n{confusion_matrix(y_test, y_pred)}')

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train.values.reshape(-1, 1), y_train)
y_predrf = rf.predict(X_test.values.reshape(-1, 1))

from sklearn.metrics import confusion_matrix, classification_report
print(f'Classification report after RF: \n{classification_report(y_test, y_predrf)}')
print(f'Confusion matrix after RF: \n{confusion_matrix(y_test, y_predrf)}')


# ## 5. Let's try to add Levenstein

# In[ ]:


def lev_dist(a1, a2):
    source=a1.split()
    target=a2.split()
    if source == target:
        return 0


    # Prepare a matrix
    slen, tlen = len(source), len(target)
    dist = [[0 for i in range(tlen+1)] for x in range(slen+1)]
    for i in range(slen+1):
        dist[i][0] = i
    for j in range(tlen+1):
        dist[0][j] = j

    # Counting distance, here is my function
    for i in range(slen):
        for j in range(tlen):
            cost = 0 if source[i] == target[j] else 1
            dist[i+1][j+1] = min(
                            dist[i][j+1] + 1,   # deletion
                            dist[i+1][j] + 1,   # insertion
                            dist[i][j] + cost   # substitution
                        )
    return (dist[-1][-1])/           ((len(source)+len(target))/2)



#df = pd.read_csv('preprocess_all.csv').dropna()
df['lev_dist'] = np.vectorize(lev_dist)(df['preprocessed_q1'], df['preprocessed_q2'])
df.to_csv('preprocess_all_lev.csv')
df['lev_pred']=0
df['lev_pred']=df.lev_dist < 0.35
df["lev_pred"]=df["lev_pred"].astype(int)
df['lev_true']=df.lev_pred ==df.is_duplicate
print(df['lev_true'].value_counts())


print(len(df))


# In[12]:


# transforming that 10000 rows
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
import time

pca_tsne = Pipeline([
    ("tsvd", TruncatedSVD(n_components=1500, random_state=42)),
    ("tsne", TSNE(n_components=3, random_state=42)),
])

tsvd = TruncatedSVD(n_components=1500, random_state=42)
t0 = time.time()
df_clean_vectorized_cv_q1_reduced = tsvd.fit_transform(df_clean_vectorized_cv_q1)
df_clean_vectorized_cv_q2_reduced = tsvd.transform(df_clean_vectorized_cv_q2)
t1 = time.time()
print(f'Our new data will have shape {df_clean_vectorized_cv_q2_reduced.shape}')
print(f"Reduction in pipeline with PCA and t-SNE took {round(t1-t0, 2)}s")


# In[72]:


df_clean.head()


# In[ ]:




