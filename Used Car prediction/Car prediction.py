#!/usr/bin/env python
# coding: utf-8

# In[217]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[218]:


df = pd.read_csv('Car details v3.csv')


# In[219]:


df.head()


# In[220]:


df.seller_type.unique()


# In[221]:


CompanyName = df['name'].apply(lambda x : x.split(' ')[0])
df.insert(1,"CompanyName",CompanyName)
df.drop('name',axis=1,inplace=True)
df.head()

Owner = df['owner'].apply(lambda x : x.split(' ')[0])
df.insert(8,"Owner",Owner)
df.drop('owner',axis=1,inplace=True)
df.head()

from datetime import date
current_date = date.today()
current_year = current_date.year


# In[222]:


df.head()


# In[223]:


from datetime import date
current_date = date.today()
current_year = current_date.year
for i in range(len(df['year'])):
    df['year'][i] = current_year - df['year'][i]


# In[224]:


df.insert(3,'cost_price',"")
for i in range(len(df['year'])):
    if df['year'][i]<=2:
        df['cost_price'][i] = int(df['selling_price'][i] *1.25)
    if 2<df['year'][i]<=4:
        df['cost_price'][i] = int(df['selling_price'][i] *1.66)
    if 4<df['year'][i]<=6:
         df['cost_price'][i] = int(df['selling_price'][i] *2)
    if df['year'][i]>6:
        df['cost_price'][i] = int(df['selling_price'][i] *2.5)


# In[225]:


plt.figure(figsize=(10,8))
sns.lmplot(x='mileage',y='year',data=df)


# In[226]:


pd.DataFrame(df.groupby(['CompanyName'])['selling_price'].mean()).plot(kind='bar',figsize=(10,8))


# In[227]:


pd.DataFrame(df.groupby(['year'])['selling_price'].mean()).plot(kind='line',figsize=(10,8))


# If the car is produced more recently the its price is also more

# In[228]:


pd.DataFrame(df.groupby(['km_driven'])['selling_price'].mean()).plot(kind='line',figsize=(10,8))


# as number of km driven increases selling price decreases

# In[229]:


plt.figure(figsize=(15,6))

plt.subplot(1,2,1)
sns.barplot(x=df['fuel'],y=df['selling_price'])
plt.title('fuel vs selling price')

plt.subplot(1,2,2)
sns.barplot(x=df['transmission'],y=df['selling_price'])
plt.title('transmission vs selling price')


# > diesel power cars tent to have high selling price
# 
# > Similarly automatic cars have higher selling price

# In[230]:


plt.figure(figsize=(19,6))

plt.subplot(1,2,1)
sns.barplot(x=df['seller_type'],y=df['selling_price'])
plt.title('seller_type vs selling price')

plt.subplot(1,2,2)
sns.barplot(x=df['Owner'],y=df['selling_price'])
plt.title('owner vs selling price')


# > if you try to sell your car through a dealer selling price increases
# 
# >similaring test drive cars have a good seeling price

# In[231]:


plt.figure(figsize=(100,50))
sns.barplot(x=df['mileage'],y=df['engine'])


# if u zoom in and look carefully as the engine's CC increases mileage decreases

# In[232]:


sns.stripplot(x=df['seats'],y=df['selling_price'])


# # Data Cleaning

# In[233]:


df.isna().sum()


# In[234]:


df.shape


# In[235]:


df.drop(['torque'],axis=1,inplace=True)


# In[236]:


df.dropna(inplace=True)


# In[237]:


df.isna().sum()


# In[238]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(df['CompanyName'])
df['CompanyName'] = le.transform(df['CompanyName'])

le.fit(df['fuel'])
df['fuel'] = le.transform(df['fuel'])

le.fit(df['seller_type'])
df['seller_type'] = le.transform(df['seller_type'])

le.fit(df['transmission'])
df['transmission'] = le.transform(df['transmission'])

le.fit(df['Owner'])
df['Owner'] = le.transform(df['Owner'])


# In[239]:


df.head()


# # Model Building

# In[240]:


x = df[['year','cost_price','km_driven','fuel','seller_type','transmission','Owner']]
y = df['selling_price']


# In[241]:


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.7,random_state=101)


# In[242]:


from sklearn.metrics import accuracy_score,confusion_matrix

from sklearn.neighbors import KNeighborsClassifier

kn = KNeighborsClassifier(n_neighbors=8)
kn.fit(x_train,y_train)
kn_predict = kn.predict(x_test)


# In[243]:


accuracy_score(y_test,kn_predict)*100


# 1)Decision Tree

# In[259]:


from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)
dt_predict = dt.predict(x_test)


# In[260]:


accuracy_score(y_test,dt_predict)*100


# 2) Random Forest 

# In[261]:


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(x_train,y_train)
rf_predict = rf.predict(x_test)


# In[262]:


accuracy_score(y_test,rf_predict)*100


# 3) support vector machine

# In[263]:


from sklearn.svm import SVC

svm = SVC()
svm.fit(x_train,y_train)
svm_predict = svm.predict(x_test)


# In[264]:


accuracy_score(y_test,svm_predict)*100


# 4) naive Bayes

# In[265]:


from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
nb.fit(x_train,y_train)
nb_predict = nb.predict(x_test)


# In[266]:


accuracy_score(y_test,nb_predict)*100


# Looks like Decision tree has done a great job then others

# In[ ]:




