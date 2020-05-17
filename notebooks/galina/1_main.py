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


data_folder = "../../data/raw/"


# In[5]:


get_ipython().system('ls "../../data/raw/"')


# In[6]:


df = pd.read_csv(os.path.join(data_folder, "sample_submission.csv.zip"), compression="zip")


# In[7]:


df.head()

