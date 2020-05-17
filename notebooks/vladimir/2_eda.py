#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
from collections import Counter
from itertools import chain, tee
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'
import swifter
import logging
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)
import warnings
warnings.filterwarnings('ignore')


# In[3]:


import os
import sys
sys.path.append(os.path.abspath('../../'))
from nlp.tokenizer import *


# In[4]:


filename = "../maria/train_dup.csv"


# In[5]:


get_ipython().system('wc -l {filename}')
get_ipython().system('wc -w {filename}')


# In[6]:


def size_iter(iterable):
    return len(list(iterable))


# In[6]:


df = pd.read_csv(filename).drop_duplicates().dropna()


# ### count words

# In[7]:


def count_words(col):
    return list(t.count() for t in SpacyTokens(list(col.values)))


# In[ ]:


df["q1_all"] = count_words(df['question1'])
print("that's all")


# In[ ]:


df["q2_all"] = count_words(df['question2'])
print("that's all")


# In[ ]:


df.to_csv(filename)


# ### count words without punct and stop words

# In[ ]:


def count_without_punct_stop_words(col):
    return list(t.remove_all(punct, stop).count() for t in SpacyTokens(list(col.values)))


# In[ ]:


df["q1_ps"] = count_without_punct_stop_words(df['question1'])
df["q2_ps"] = count_without_punct_stop_words(df['question2'])
print("that's all")


# In[ ]:


df.to_csv(filename)


# ### creating questions words

# In[42]:



questions = {"who":1<<1, "whom":1<<2, "whose":1<<3, "what":1<<4, "when":1<<5, "where":1<<6, "why":1<<7, "how":1<<8,
                                    "there":1<<9, "that":1<<10, "which":1<<11, "whither":1<<12, "whence":1<<13, "whether":1<<14, "whatsoever":1<<15}

def question_words(col):
    return list(list(t.filter(question)) for t in SpacyTokens(list(col.values)))

def get_questions(q):
    return sum(SpacyTokens(q).lower().map(lambda t:questions.get(t.text.lower(), 0)))


def create_wh_ds(df, target_column, out_column, filename):
    dfw = df[[target_column]][:].reset_index(drop=True)
    dfw[out_column] = np.vectorize(get_questions)(dfw[target_column])
    for q, mask in questions.items():
        dfw[q] = (np.bitwise_and(dfw[out_column], mask)!=0).astype(int)
    dfw.to_csv(filename)

create_wh_ds(df, "question1", "wh1", "./q1_question_word.csv")
create_wh_ds(df, "question2", "wh2", "./q2_question_word.csv")
print("that's all")


# ### create freq dict

# In[18]:


def get_tokens(col):
    d = {}
    for t in SpacyTokens(list(col.values)).lower().flatten():
        d[t.text] = d.get(t.text,0) + 1
    return d


# In[21]:


d_freq1 = get_tokens(df["question1"])
utils.to_pickle(d_freq1, "./question1_freq.pkl")
print("that's all")


# In[20]:


d_freq2 = get_tokens(df["question2"])
utils.to_pickle(d_freq2, "./question2_freq.pkl")
print("that's all")


# In[39]:


df1_freq = pd.DataFrame.from_dict(d_freq1, orient='index', columns=["count"]).reset_index()
df1_freq.columns = ["word","count"]
df2_freq = pd.DataFrame.from_dict(d_freq2, orient='index', columns=["count"]).reset_index()
df2_freq.columns = ["word","count"]


# In[42]:


def get_token(q):
    return next(SpacyTokens(q))
tokens1 = np.vectorize(get_token)(df1_freq["word"].values)
tokens2 = np.vectorize(get_token)(df2_freq["word"].values)


# In[58]:


df1_freq["token"] = tokens1
df2_freq["token"] = tokens2


df1_freq["is_digit"] = df1_freq["token"].apply(lambda t:t.is_digit)
df1_freq["is_oov"] = df1_freq["token"].apply(lambda t:t.is_oov)
df1_freq["is_punct"] = df1_freq["token"].apply(lambda t:t.is_punct)
df1_freq["is_stop"] = df1_freq["token"].apply(lambda t:t.is_stop)

df2_freq["is_digit"] = df2_freq["token"].apply(lambda t:t.is_digit)
df2_freq["is_oov"] = df2_freq["token"].apply(lambda t:t.is_oov)
df2_freq["is_punct"] = df2_freq["token"].apply(lambda t:t.is_punct)
df2_freq["is_stop"] = df2_freq["token"].apply(lambda t:t.is_stop)

df1_freq = df1_freq.drop("token", axis=1)
df2_freq = df2_freq.drop("token", axis=1)

df1_freq.to_csv("./q1_freq.csv")
df2_freq.to_csv("./q2_freq.csv")


# In[40]:




