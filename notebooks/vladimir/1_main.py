#!/usr/bin/env python
# coding: utf-8

# In[4]:


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


# In[5]:


import os
import sys
sys.path.append(os.path.abspath('../../'))
from nlp.tokenizer import *


# In[2]:


data_folder = "../../data/raw/"


# In[7]:


get_ipython().system('ls "../"')


# In[8]:


df = pd.read_csv(
        "../maria/train_dup.csv").drop_duplicates().dropna()
corpus = pd.concat([df['question1'], df['question2']]).unique()[:100]
for t in SpacyTokens(corpus).remove(punct):
    print(t)


# In[9]:


for t in SpacyTokens(corpus).remove_all(punct, number):
    print(t)


# In[10]:


for t in SpacyTokens("I go to the supermarket").remove_all(punct, number, stop):
    print(t)


# In[ ]:




