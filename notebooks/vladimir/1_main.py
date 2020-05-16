#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'
import warnings
warnings.filterwarnings('ignore')


# In[3]:


import os
import sys
sys.path.append(os.path.abspath('../../'))
from nlp.tokenizer import *


# In[3]:


data_folder = "../../data/raw/"


# In[4]:


get_ipython().system('ls "../"')


# In[5]:


df = pd.read_csv(
        "../maria/train_dup.csv").drop_duplicates().dropna()
corpus = pd.concat([df['question1'], df['question2']]).unique()[:100]
for t in SpacyTokens(corpus).remove(punct):
    print(t)


# In[6]:


for t in SpacyTokens(corpus).remove_all(punct, number).chunk(2):
    print(t)


# In[7]:


for t in SpacyTokens("I go to the supermarket").lemmatize():
    print(t)


# In[12]:


for t in SpacyTokens("I went to the supermarket").lemmatize():
    print(t)


# In[8]:



for t in SpacyTokens("London is The Capital Of Great Britain").lower():
    print(t)


# In[9]:


for t in SpacyTokens("ipad 4").remove(regex("\d+")):
    print(t)


# In[10]:


from nltk import ngrams

for t in SpacyTokens(corpus).remove_all(punct, number).ngrams(2):
    print(t)

