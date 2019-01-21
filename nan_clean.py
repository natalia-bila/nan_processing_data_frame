#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from collections import defaultdict
from scipy.stats import hmean
from scipy.spatial.distance import cdist
from scipy import stats
import numbers


# In[13]:


def nan_detect(df):
    nan_pos = []
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            try:
                if  pd.isna(df.iloc[i,j]):             
                        nan_pos.append((i,j))
            except:
                pass
    return nan_pos


# In[14]:


def del_nan(df, col = False):
    if col:
        df = df.drop(df.columns[[i[1] for i in nan_detect(df) ]], axis=1)

    else:
        df = df.drop(df.index[[i[0] for i in nan_detect(df) ]], axis = 0)

    return df


# In[15]:


def stat_columns(df, stat = 'median'):
    columns_replace_nan = {}
    for column in df.columns:
        if (df[column].dtype  == float) or (df[column].dtype  == int):
            if stat == 'median':
                columns_replace_nan[column] = df[column].median()
            elif stat == 'mean':
                columns_replace_nan[column] = df[column].mean()
            elif stat == 'mode':
                columns_replace_nan[column] = df[column].mode()[0]
        else:
            columns_replace_nan[column] = df[column].mode()[0]
    return columns_replace_nan 


# In[2]:


def nan_to_stat(df, stat = 'median'):
    df = pd.DataFrame(df)
    columns_replace_nan = stat_columns(df, stat)
    for elem_pos in nan_detect(df):
        df.iloc[elem_pos] = columns_replace_nan[df.columns[elem_pos[1]]]
    return df


# In[17]:


def nan_to_logreg(df):
    columns_replace_nan = []
    for column in df.columns:
        if (df[column].dtype  == float) or (df[column].dtype  == int):
            columns_replace_nan.append(column)
    df_new = df[:]
    nans = [j[0] for j in nan_detect(df)]
    for i, elem in enumerate(columns_replace_nan):   
        lr = LogisticRegression()
        df_num = df[columns_replace_nan[:i]+columns_replace_nan[i+1:]]
        lr = lr.fit(df_num.drop(nans),df[elem].drop(nans))
        loc_pred = [j[0] for j in nan_detect(df) if ((df.columns[j[1]]==elem) and (pd.isna(df_num.loc[j[0]]).sum()==0))]
        if loc_pred:
            y_pred = lr.predict(df_num.loc[loc_pred])
            df_new.loc[(loc_pred, elem)] = y_pred
    return df_new


# In[18]:


def standartize(df):
    for column in df.columns:
        if (df[column].dtype  == float) or (df[column].dtype  == int):
            df[column] = (df[column]-df[column].mean())/df[column].std()
    return df


# In[19]:


def scaling(df):
    for column in df.columns:
        if (df[column].dtype  == float) or (df[column].dtype  == int):
            df[column] = (df[column]-df[column].min())/(df[column].max() - df[column].min())
    return df


# In[ ]:




