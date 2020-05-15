#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[4]:


import os
import sys
sys.path.append(os.path.abspath('../../'))
from nlp.preprocessing import spacy_tokenizer


# In[5]:


data_folder = "../../data/raw/"


# In[6]:


get_ipython().system('ls "../../data/raw/"')


# In[9]:


df = pd.read_csv(os.path.join(data_folder, "train.csv.zip"), compression="zip").set_index("id")


# In[10]:


df.head()


# In[15]:


100*df["is_duplicate"].value_counts()/len(df)


# In[ ]:




