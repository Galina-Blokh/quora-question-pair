#!/usr/bin/env python
# coding: utf-8

# In[150]:


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
import string
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


# In[99]:


# CONSTANTS
PUNCTS = string.punctuation and {'...', '?', '(', ')'} # TODO: should be improved
STOP_WORDS = set(stopwords.words("english")) - {'who', 'whom', 'what', 'when', 'where', 'why', 'how'}


# In[72]:


df = pd.read_csv('train_dup.csv').drop(columns='id', axis=1)
df = df.dropna()
df.head()


# In[73]:


duplicates = df[df.is_duplicate==1]
not_duplicates = df[df.is_duplicate == 0]
print(f'The proportion of classes pairs/not pairs is: {len(duplicates)}/{len(not_duplicates)}')
print(f'An example of duplicated questions: \n{duplicates.question1[5]} \nAND \n{duplicates.question2[5]}')
duplicates.head()


# In[33]:


# checking out from habr. This example will be later removed
def compare_stemmer_and_lemmatizer(stemmer, lemmatizer, word, pos):
    """
    Print the results of stemmind and lemmitization using the passed stemmer, lemmatizer, word and pos (part of speech)
    """
    print("Stemmer:", stemmer.stem(word))
    print("Lemmatizer:", lemmatizer.lemmatize(word, pos))
    print()

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
compare_stemmer_and_lemmatizer(stemmer, lemmatizer, word = "seen", pos = wordnet.VERB)
compare_stemmer_and_lemmatizer(stemmer, lemmatizer, word = "drove", pos = wordnet.VERB)


# In[37]:


# This is an example, will be later removed
count_vectorizer = CountVectorizer()
print(duplicates.question1[5], duplicates.question2[5])
# Create the Bag-of-Words Model
bag_of_words = count_vectorizer.fit_transform([duplicates.question1[5], duplicates.question2[5]])

# Show the Bag-of-Words Model as a pandas DataFrame
feature_names = count_vectorizer.get_feature_names()
pd.DataFrame(bag_of_words.toarray(), columns = feature_names)


# Так, мы здесь взяли 2 похожих вопроса, каждый соответствует ряду теперь. Можно заметить, что вектора типа очень близки, но нужно смотреть на ключевые слова, а не на предлоги, союзы и т.д
# 
# 
# Мы будем потом выбрасывать эти стоп-слова, которые есть в stopwords.words("english")

# # Планы:

# игнорирование регистра слов;
# 
# 
# игнорирование пунктуации;
# 
# 
# выкидывание стоп-слов;
# 
# 
# приведение слов к их базовым формам (лемматизация);
# 
# 
# исправление неправильно написанных слов.
# 
# 
# подсчет н-грамм
# 
# 
# TF/IDF - как помогут?
# 
# 
# как преобразовать в числа?

# ![title](project_work_scheme.png)

# In[63]:


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
    sentence = [lemmatizer.lemmatize(word) for word in sentence if word not in set(stopwords.words("english"))]
    sentence = " ".join(sentence)
    return sentence

print(duplicates.question1[5])
clean_sentencelemm(duplicates.question1[5])


# ## SPACY

# Tokenization, lemmatization, removing stop words (except question opening words like who, why, etc) and puctuation and lowering

# In[104]:


clean_question = lambda sentence: ' '.join([word.lemma_.lower() for word in nlp(sentence) if word not in STOP_WORDS if word.lemma_ not in PUNCTS])
df_clean = df.copy()
df_clean['question1'] = df.apply(lambda row: clean_question(row['question1']), axis=1)
df_clean['question2'] = df.apply(lambda row: clean_question(row['question2']), axis=1)
df_clean.head()


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


# In[ ]:


# let's check out TF(term) = term_frequency / sum(all terms frequences) = term_freq / |terms|

# tf_dict_initial = {unigramm : unigramm_count_dict_initial[unigramm] / sum(unigramm_count_dict_initial.values()) 
#                   for unigramm in unigramm_count_dict_initial}
# tf_dict_clean = {unigramm : unigramm_count_dict_clean[unigramm] / sum(unigramm_count_dict_clean.values())
#                 for unigramm in unigramm_count_dict_clean}

# now it's time for IDF(term, documnet_with_all_questions) = log ( 2 * len(df) / |questions containing term|)

questions_with_term_initial = [math.log( 2*len(df) / (sum([int(unigramm in question) for question in 
                    list(df.question1) + list( df.question2)])), 2) for unigramm in unigramm_count_dict_initial]

idf_dict_initial = dict(zip(unigramm_count_dict_initial, questions_with_term_initial))


# In[ ]:


# let's check out TF(term) = term_frequency / sum(all terms frequences)

tf_dict_initial = {unigramm : unigramm_count_dict_initial[unigramm] / sum(unigramm_count_dict_initial.values()) 
                  for unigramm in unigramm_count_dict_initial}
tf_dict_clean = {unigramm : unigramm_count_dict_clean[unigramm] / sum(unigramm_count_dict_clean.values())
                for unigramm in unigramm_count_dict_clean}

# now it's time for IDF(term, documnet_with_all_questions) = log ( 2 * len(df) / |questions containing term|)

questions_with_term_initial = [math.log( 2*len(df) / (sum([int(unigramm in question) for question in 
                    list(df.question1) + list( df.question2)])), 2) for unigramm in unigramm_count_dict_initial]

idf_dict_initial = dict(zip(unigramm_count_dict_initial, questions_with_term_initial))

questions_with_term_clean = [math.log( 2*len(df_clean) / (sum([int(unigramm in question) for question in 
        list(df_clean.question1) + list( df_clean.question2)])), 2) for unigramm in unigramm_count_dict_clean]

idf_dict_clean = dict(zip(unigramm_count_dict_clean, questions_with_term_clean))


# In[ ]:




