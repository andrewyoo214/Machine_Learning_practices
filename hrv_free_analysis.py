#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score


# In[3]:


# HRV 데이터셋 불러오기
hrv_df = pd.read_csv('E:/RESEARCH/Datasets/HRV_samsung/HRV_REV_all.csv', sep=',')
hrv_df.head()


# In[11]:


# Checking the correlation between variables
print(hrv_df.corr())
plt.figure(figsize=(20, 20))
sns.heatmap(hrv_df.corr())
plt.show()


# In[32]:


## Assigning X and Y values

# data columns preprocessing by slicing
hrv_variables = hrv_df.drop(['sub','disorder','HAMD', 'HAMA','PDSS','ASI','APPQ','PSWQ','SPI','PSS','BIS','SSI' ], axis=1)
hrv_target = hrv_df.loc[:,['disorder']]
hrv_patient = hrv_df.loc[:, ['sub']]
hrv_HAMD = hrv_df.loc[:,['HAMD']]
hrv_HAMA = hrv_df.loc[:,['HAMA']]
hrv_PDSS = hrv_df.loc[:,['PDSS']]
hrv_ASI = hrv_df.loc[:,['ASI']]
hrv_APPQ = hrv_df.loc[:,['APPQ']]
hrv_PSWQ = hrv_df.loc[:,['PSWQ']]
hrv_SPI = hrv_df.loc[:,['SPI']]
hrv_PSS = hrv_df.loc[:,['PSS']]
hrv_BIS = hrv_df.loc[:,['BIS']]
hrv_SSI = hrv_df.loc[:,['SSI']]

# assign values
x = hrv_variables.values
y = hrv_target.values
p = hrv_patient.values


# In[29]:


## splitting the dataset into training set and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


# In[30]:


# feature scaling. set values between 0 and 1
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)


# In[25]:


## KNN Classifier

# from sklearn.neighbors import KNeighborsClassifier
# parameters = {'n_neighbors': list(range(0,51)), 'metric':['minkowski', 'euclidean', 'manhattan', 'chebyshev']}
# grid_search = GridSearchCV(estimator = KNeighborsClassifier(), param_grid = parameters, scoring = 'accuracy', cv = 10, n_jobs = -1)
# grid_search.fit(x_train, y_train)

# print('KNN Classifier Grid Search Best Accuracy = {:.2f}%'.format(grid_search.best_score_ *100))
# print('KNN Classifier Best Parameters:', grid_search.best_params_)


# In[31]:


from sklearn.linear_model import LogisticRegression
C = list(range(1,11))
parameters = {'C': C, 'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}
grid_search = GridSearchCV(estimator = LogisticRegression(random_state = 0), param_grid = parameters, scoring = 'accuracy', cv = 10, n_jobs = -1)
grid_search.fit(x_train, y_train)

print('Logistic Regression Grid Search Best Accuracy = {:.2f}%'.format(grid_search.best_score_ *100))
print('Logistic Regression Best Parameters:', grid_search.best_params_)


# In[ ]:




