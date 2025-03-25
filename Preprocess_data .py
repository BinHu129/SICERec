#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.io import loadmat


# In[3]:


click_f = loadmat('data/epinions/rating.mat')['rating']
trust_f = loadmat('data/epinions/trustnetwork.mat')['trustnetwork']


# In[5]:


click_list=[]


# In[6]:


for x in tqdm(click_f):
    uid = x[0]
    iid = x[1]
    species=x[2]
    label = x[3]
    click_list.append([uid, iid,species,label])


# In[7]:


pos_list = []
for x in click_list:
	pos_list.append((x[0], x[1],x[2],x[3]))
    
pos_list = list(set(pos_list))
random.shuffle(pos_list)
num_test = int(len(pos_list) * 0.1)
test_set = pos_list[:num_test]
valid_set = pos_list[num_test:2 * num_test]
train_set = pos_list[2 * num_test:]
print('Train samples: {}, Valid samples: {}, Test samples: {}'.format(len(train_set), len(valid_set), len(test_set)))


# In[8]:


with open('data/trained models epinions/dataset_epinions.pkl', 'wb') as f:
	pickle.dump(train_set, f, pickle.HIGHEST_PROTOCOL)
	pickle.dump(valid_set, f, pickle.HIGHEST_PROTOCOL)
	pickle.dump(test_set, f, pickle.HIGHEST_PROTOCOL)


# In[ ]:




