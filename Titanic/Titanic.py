#!/usr/bin/env python
# coding: utf-8

# In[75]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[76]:


df = pd.read_csv('titanic.csv')


# In[77]:


df.head()


# In[135]:


df.corr().mean()


# In[78]:


sns.countplot(x='Survived', data=df)


# In[79]:


## Let's check who are with family and who are alone
## This can be found by adding Parch and Sibsp columns
df['Alone'] = df.Parch + df.SibSp

## if Alone value is >0 then they are with family else they are Alone
df['Alone'].loc[df['Alone']>0] = 'With Family'
df['Alone'].loc[df['Alone'] == 0] = 'Without Family'

sns.countplot(x=df['Alone'])


# In[80]:


df['Sex'].value_counts().plot(kind='pie')


# In[81]:


df['Pclass'].value_counts().plot(kind='pie')


# In[82]:


plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.countplot(x=df['Survived'],hue=df['Sex'])
plt.title('Survival ratio based on sex')

plt.subplot(1,2,2)
sns.countplot(x=df['Survived'],hue=df['Pclass'])
plt.title('Survival ratio based on class')


# # Data Cleaning

# In[83]:


df.isna().sum()


# In[84]:


df.drop(['Cabin'],axis=1,inplace=True)


# In[85]:


df.dropna(inplace=True)


# In[86]:


df.isna().sum()


# In[87]:


df.drop(['PassengerId','Name','Ticket','Fare','Alone'],axis=1,inplace=True)


# In[88]:


df.head()


# In[89]:


from sklearn.preprocessing import LabelEncoder


# In[90]:


le = LabelEncoder()
le.fit(df['Sex'])
df['Sex'] = le.transform(df['Sex'])

le.fit(df['Embarked'])
df['Embarked'] = le.transform(df['Embarked'])


# # Traning and test

# In[91]:


from sklearn.model_selection import train_test_split


# In[136]:


x = df.drop(['Survived'],axis=1)
y = df['Survived']

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.6,random_state=101)


# # Model Building
# 

# 1)Logistic Regression

# In[137]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix

lg = LogisticRegression(max_iter=450)
lg.fit(x_train,y_train)
lg_predict = lg.predict(x_test)
lg_cm = confusion_matrix(lg_predict,y_test)
sns.heatmap(lg_cm,annot=True)


# In[138]:


accuracy_score(y_test,lg_predict)


# 2) K-Nearest Neighbor

# In[139]:


from sklearn.neighbors import KNeighborsClassifier

kn = KNeighborsClassifier(n_neighbors=5)
kn.fit(x_train,y_train)
kn_predict = kn.predict(x_test)
kn_cm = confusion_matrix(y_test,kn_predict)
sns.heatmap(kn_cm,annot=True)


# In[140]:


accuracy_score(y_test,kn_predict)


# 3)Decision Tree

# In[141]:


from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)
dt_predict = dt.predict(x_test)
dt_cm = confusion_matrix(y_test,dt_predict)
sns.heatmap(dt_cm,annot=True)


# In[142]:


accuracy_score(y_test,dt_predict)


# 4)Random Forest

# In[143]:


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(x_train,y_train)
rf_predict = rf.predict(x_test)
rf_cm = confusion_matrix(y_test,rf_predict)
sns.heatmap(rf_cm,annot=True)


# In[144]:


accuracy_score(y_test,rf_predict)


# 5)Support Vector Machine

# In[145]:


from sklearn.svm import SVC

svm = SVC()
svm.fit(x_train,y_train)
svm_predict = svm.predict(x_test)
svm_cm = confusion_matrix(y_test,svm_predict)
sns.heatmap(svm_cm,annot=True)


# In[146]:


accuracy_score(y_test,svm_predict)


# So in all above Model LogisticRegression has the highest accuracy of 82%

# In[ ]:




