#!/usr/bin/env python
# coding: utf-8

# In[105]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[106]:


df = pd.read_csv('matches.csv')


# In[107]:


df.head()


# In[202]:


plt.figure(figsize=(15,6))
win = df.winner.value_counts().reset_index()
sns.barplot(x=win['index'],y=win['winner'],palette='RdYlGn_r')
plt.title('Most Matches Won')
plt.xticks(rotation=90)


# According to the chart above most successful teams are as follows
# 
# **1) Mumbai Indians**
# 
# **2) Chennai Super Kings**
# 
# **3) Kolkata Knight Riders**

# In[203]:


plt.figure(figsize=(15,6))
pom = df.player_of_match.value_counts()[:20].reset_index()
sns.barplot(x=pom['player_of_match'],y=pom['index'],palette='summer')
plt.title('Most Player of the Match')


# Players with most man of the match for their outstand performances 
# 
# **1) CH Gayle**
# 
# **2) AB de Villiers**
# 
# **3) MS Dhoni**

# In[204]:


plt.figure(figsize=(10,8))
tos = df.toss_winner.value_counts().reset_index()
sns.barplot(x=tos['toss_winner'],y=tos['index'],palette='Set3')
plt.title('Most Tosses Won By')


# Team that have won the toss most numbers of time
# 
# **1) Mumbai Indians**
# 
# **2) Kolkata Knight Riders**
# 
# **3) Chennai Super Kings**

# In[205]:


plt.figure(figsize=(15,6))
sns.countplot(x=df['toss_winner'],hue=df['toss_decision'])
plt.xticks(rotation=90)
plt.title('Comparision Between Choice of Field and bat')


# In[112]:


tos = df.toss_decision.value_counts().reset_index()
plt.pie(tos['toss_decision'],labels=tos['index'])


# > from the chart above we find that most of the teams tend to choose Field after winning the toss and try to chase the total 
# 
# > Chennai super kings is the exception in this case the tend to choose to bat and the defend the total

# In[113]:


df1 = df[df['win_by_runs']!=0]
print("Number of times Match was won team defending the target",len(df1))


# **Wins While Defending = 337**

# In[114]:


df2 = df[df['win_by_wickets']!=0]
print("Number of times match was won by the team chasing down the total",len(df2))


# **Wins while chasing = 406**

# In[115]:


print("Number of times match was tied",len(df2)-len(df1))


# **ties = 69**

# In[198]:


season_winner = df.drop_duplicates(subset=['season'],keep='last')[['season', 'winner']].reset_index(drop=True)
season_winner.winner.value_counts().plot(kind='barh',orientation='horizontal',title='Most Title wins')


# **Mumbai Indians** is most successful team with **4 ipl titles** followed by **Chennai super Kings** with **3 titles**

# # Player Records

# In[163]:


dff = pd.read_csv('deliveries.csv')


# In[164]:


dff.head()


# In[165]:


dff.info()


# # Batting Records

# In[206]:


plt.figure(figsize=(15,6))
dff2 = dff[dff['batsman_runs']==6]
bat = dff2.batsman.value_counts()[:20].reset_index()
sns.barplot(x=bat['index'],y=bat['batsman'],palette='RdYlGn')
plt.xticks(rotation=90)
plt.title('Most Sixes')


# **CH Gayle** is most explosive batsman with over **300 6's** followed by **Ab de Villers** with over **200 6's** followed bt **MS Dhoni** with just a little more then **200 6's** 

# In[207]:


plt.figure(figsize=(15,6))
dff2 = dff[dff['batsman_runs']==4]
bat = dff2.batsman.value_counts()[:20].reset_index()
sns.barplot(x=bat['index'],y=bat['batsman'],palette='RdYlGn')
plt.xticks(rotation=90)
plt.title('MOst Fours')


# Talking about 4's **S Dhawan** Tops the list with over **500 4's** followed by **Sk Raina and G.Gambhir** with almost **500 4's**

# In[208]:


batsmen = dff.groupby("batsman").agg({'ball': 'count','batsman_runs': 'sum'})
batsmen.rename(columns={'ball':'balls', 'batsman_runs': 'runs'}, inplace=True)
batsmen = batsmen_summary.sort_values(['balls','runs'], ascending=False)[:10]
plt.figure(figsize=(15,6))
sns.barplot(x=batsmen.index,y=batsmen['runs'])
plt.xticks(rotation=90)
plt.title('Most Runs')


# **V Kohil** is the highest scorer in the whole tournament with above **5000 runs**
# 
# **Sk Raina** is next in line with V Kohil with just few runs short of him
# 
# **Rg Sharma** is next in line with almost **5000 runs**

# In[209]:


batsmen1 = batsmen
batsmen1['batting_strike_rate'] = batsmen1['runs']/batsmen1['balls'] * 100
batsmen1['batting_strike_rate'] = batsmen1['batting_strike_rate'].round(2)
batsmen1 = batsmen1[:10]
plt.figure(figsize=(15,6))
sns.barplot(x=batsmen1.index,y=batsmen['batting_strike_rate'])
plt.xticks(rotation=90)
plt.title('Best Strike Rate')


# In T20 format strike rate is very important part of the game 
# 
# **CH Gayle, DA Warner and MS Dhoni** top the list with strike rate of around **140**

# # Bowling Records

# In[210]:


bowler = dff.groupby('bowler').agg({'ball':'count','total_runs':'sum','player_dismissed':'count'})
bowler.rename(columns = {'ball':'balls','total_runs':'runs','player_dismissed':'wickets'},inplace=True)
bowler=bowler.sort_values(['wickets'],ascending=False)[:20]
plt.figure(figsize=(15,6))
sns.barplot(x=bowler.index,y=bowler['wickets'],palette='RdYlGn')
plt.xticks(rotation=90)
plt.title('Most Wickets')


# > **SL Malinga** is most destructive bowler with **over 175 wickets** 
# 
# > Next is **Dj Bravo** with **almost 170 wickets** in his pocket
# 
# > **A Mishra** is next in line with **almost 170 wickets** but few wickets short to beat Dj Bravo

# In[211]:


bowler['economy'] = bowler['runs']/(bowler['balls']/6)
bowler = bowler.sort_values(['economy'],ascending=True)
plt.figure(figsize=(15,6))
sns.barplot(x=bowler.index,y=bowler['economy'],palette='RdYlGn_r')
plt.xticks(rotation=90)
plt.title('Most Economical Bowler')


# Along with number of wickets bowling economy matters a lot in a bowlers history
# 
# 1) **DW Steyn** has lowest bowling economy of around **6.7runs**
# 
# 2) Next is **R Ashwin** with bowling economy of around **6.8runs**
# 
# 3) next is **SP Narine** with economy of approx **7.0runs**

# **After Detail analysis on the data we have found that **
# 
# 1) **Mumbai Indians** is Most Successful team in the history of tournament 
# 
# 2) **V Kohil** is most destructive batsmen 
# 
# 3) **Sl Malinga** is most fearced Bowler**
# 

# According to my analysis i will suggest that
# 1) **Mumbai Indians, Chennai Super Kings and Kolkata Knight Riders** are the top notch team to endorse
# 
# 2) **V Kolhi, RG Sharma, CH Gayle, AB de Villers** are some of the best batsman to endorse
# 
# 3) **SL Malinga, DW Steyn, DJ Bravo, A Mishra** are few best Bowlers to endorse

# In[ ]:




