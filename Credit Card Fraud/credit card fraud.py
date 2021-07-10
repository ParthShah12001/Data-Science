#!/usr/bin/env python
# coding: utf-8

# # Data Visualization

# In[117]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[118]:


df = pd.read_csv('creditcardcsvpresent.csv')


# In[119]:


df.head(10)


# In[120]:


df.tail(10)


# In[121]:


df.shape


# In[122]:


df['isFradulent'].value_counts().plot(kind='pie')


# In[123]:


sns.kdeplot(x=df['Average Amount/transaction/day'],hue=df['isFradulent'])


# In[124]:


sns.countplot(x=df['isForeignTransaction'],hue=df['isFradulent'])


# The number of foreign transaction which have took place among them most of them are fraud

# In[125]:


sns.stripplot(y=df['Transaction_amount'],x=df['isFradulent'])


# Looks like big transaction amount are more likely to be a fraud

# In[126]:


plt.figure(figsize=(15,8))

plt.subplot(1,2,1)
sns.lineplot(x=df['Is declined'],y=df['6-month_chbk_freq'])

plt.subplot(1,2,2)
sns.lineplot(x=df['Is declined'],y=df['isHighRiskCountry'])


# > as number of decline increases number of charge back also increases
# 
# > Similarly is transaction is made to a high risk country chance of getting declined also increases

# In[127]:


plt.figure(figsize=(15,10))

plt.subplot(1,2,1)
sns.countplot(x=df['Total Number of declines/day'],hue=df['isFradulent'])

plt.subplot(1,2,2)
sns.countplot(x=df['6-month_chbk_freq'],hue=df['isFradulent'])


# > if we look carefully as the total number of declines increases the chances of being fradulent increases
# 
# > Similarly is frequency of charge_back increases then changes of being Fraud also increases

# # Data Cleaning

# In[128]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(df['isFradulent'])
df['isFradulent'] = le.transform(df['isFradulent'])

le.fit(df['isHighRiskCountry'])
df['isHighRiskCountry'] = le.transform(df['isHighRiskCountry'])

le.fit(df['Is declined'])
df['Is declined'] = le.transform(df['Is declined'])

le.fit(df['isForeignTransaction'])
df['isForeignTransaction'] = le.transform(df['isForeignTransaction'])


# In[130]:


df.drop(['Merchant_id','Transaction date'],axis=1,inplace=True)


# In[131]:


from sklearn.utils import resample,shuffle

zero =df[df['isFradulent']==0]
one = df[df['isFradulent']==1]

upsampled1 = resample(one, replace=True, n_samples=zero.shape[0])

df = pd.concat([zero,upsampled1])

df = shuffle(df)


# In[133]:


sns.countplot(x=df['isFradulent'])


# # Model Building

# In[134]:


x = df.drop(['isFradulent'],axis=1)
y = df['isFradulent']


# In[135]:


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.7,random_state=101)


# 1)Logistic regression

# In[136]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix

lg = LogisticRegression(max_iter=450)
lg.fit(x_train,y_train)
lg_predict = lg.predict(x_test)
lg_cm = confusion_matrix(lg_predict,y_test)
sns.heatmap(lg_cm,annot=True)


# In[137]:


accuracy_score(y_test,lg_predict)*100


# 2)K-Nearset Neighbours

# In[146]:


from sklearn.neighbors import KNeighborsClassifier

kn = KNeighborsClassifier(n_neighbors=5)
kn.fit(x_train,y_train)
kn_predict = kn.predict(x_test)
kn_cm = confusion_matrix(y_test,kn_predict)
sns.heatmap(kn_cm,annot=True)


# In[147]:


accuracy_score(y_test,kn_predict)*100


# 3)Decision Tree

# In[148]:


from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)
dt_predict = dt.predict(x_test)
dt_cm = confusion_matrix(y_test,dt_predict)
sns.heatmap(dt_cm,annot=True)


# In[149]:


accuracy_score(y_test,dt_predict)*100


# 4)Random Forest

# In[150]:


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(x_train,y_train)
rf_predict = rf.predict(x_test)
rf_cm = confusion_matrix(y_test,rf_predict)
sns.heatmap(rf_cm,annot=True)


# In[151]:


accuracy_score(y_test,rf_predict)*100


# Support vector machine

# In[152]:


from sklearn.svm import SVC

svm = SVC()
svm.fit(x_train,y_train)
svm_predict = svm.predict(x_test)
svm_cm = confusion_matrix(y_test,svm_predict)
sns.heatmap(svm_cm,annot=True)


# In[153]:


accuracy_score(y_test,svm_predict)*100


# Decison Tree has the heighest accuracy of 98.92%

# In[ ]:




