#!/usr/bin/env python
# coding: utf-8

# ### HRV data preprocessing for FL research

# In[56]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image


# In[33]:


# HRV 데이터셋 불러오기
hrv_df = pd.read_csv('E:/RESEARCH/Datasets/HRV_samsung/HRV_REV_all.csv', sep=',')
# HRV_REV_visit1to5를 불러왔는데, 이는 이미 visit 1에서 5를 모두 가지고 있는 데이터들만 추려낸 것.
hrv_df.head()


# In[34]:


hrv_df.shape


# In[35]:


hrv_df['null1'] = 0
hrv_df['null2'] = 0
hrv_df['null3'] = 0
hrv_df['null4'] = 0
hrv_df['null5'] = 0
hrv_df['null6'] = 0
hrv_df['null7'] = 0
hrv_df['null8'] = 0
hrv_df['null9'] = 0
hrv_df['null10'] = 0


# In[36]:


hrv_df.shape


# In[37]:


hrv_df.head()


# In[7]:


#HRV 데이터셋에서 VISIT1, 즉 첫번째 방문에 대한 데이터만을 hrv_visit1에 저장
# hrv_visit1=hrv_df[hrv_df['VISIT']==1]
# hrv_visit1.head(10)


# In[47]:


#disorder값은 pixel에 넣지 않음. 
hrv_100 = hrv_df.drop(['sub','disorder','VISIT'], axis=1)
hrv_100.head()


# ##### 이제 총 100개의 column으로 구성되었으니까 normalization 하자

# In[49]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
hrv_100[:] = scaler.fit_transform(hrv_100[:])


# In[50]:


hrv_100.head()


# In[51]:


hrv_100.shape


# In[52]:


x0=hrv_100.loc[0].values


# In[53]:


x0


# In[54]:


x0=x0.reshape(10,10)


# In[55]:


x0 = sns.heatmap(x0)


# In[61]:


from keras.preprocessing.image import array_to_img
x0=array_to_img(x0)


