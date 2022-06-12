#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,roc_curve,roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')


# In[2]:


data=pd.read_csv('https://raw.githubusercontent.com/dsrscientist/dataset1/master/abalone.csv')
data.head()


# In[3]:


data.rename(columns = {'Whole weight':'w_weight','Shucked weight':'shuck_weight','Viscera weight':'v_weight','Shell weight':'shell_weight' }, inplace = True)
data


# In[4]:


data['age'] = data['Rings']+1.5
data.drop('Rings', axis = 1, inplace = True)
data


# In[5]:


data.shape


# In[6]:


data.isna().sum()


# In[7]:


label_enc=LabelEncoder()


# In[8]:


df1=label_enc.fit_transform(data['Sex'])
df1


# In[9]:


data['Sex']=df1
data


# In[10]:


data.describe()


# In[11]:


plt.figure(figsize=(20,15),facecolor="yellow")
plotnumber=1
for column in data:
    if plotnumber<=9:
        ax=plt.subplot(3,3,plotnumber)
        sns.distplot(data[column])
        plt.xlabel(column,fontsize=10)
    plotnumber+=1
plt.show()


# In[12]:


data['age'].mean()


# In[13]:


Age = []
for i in data['age']:
    if i > 11.43:
        Age.append('1')
    else:
        Age.append('0')
data['age']=Age
data


# In[14]:


#Now, let's split our data into features and label
x=data.drop(columns="age", axis=1)
y=data["age"]


# In[15]:


y


# In[16]:


plt.figure(figsize=(20,50))
graph=1
for column in x:
    if plotnumber<=50:
        ax=plt.subplot(10,5,graph)
        sns.boxplot(data=data[column])
        plt.xlabel(column,fontsize=20)
    graph+=1
plt.show()


# In[17]:


#1st Quantile
q1=data.quantile(0.25)

#3rd Quantile
q3=data.quantile(0.75)

iqr=q3-q1


# In[18]:


length_low=(q1.Length-(1.5*iqr.Length))
index=np.where(data['Length']<length_low)
data=data.drop(data.index[index])
data.reset_index()


# In[19]:


Dia_low=(q1.Diameter-(1.5*iqr.Diameter))
index=np.where(data['Diameter']<Dia_low)
data=data.drop(data.index[index])
data.reset_index()


# In[20]:


Height_low=(q1.Height-(1.5*iqr.Height))
index=np.where(data['Height']<Height_low)
data=data.drop(data.index[index])
data.reset_index()


# In[21]:


Height_high=(q3.Height+(1.5*iqr.Height))
index=np.where(data['Height']>Height_high)
data=data.drop(data.index[index])
data.reset_index()


# In[22]:


whole_weight_high=(q3.w_weight+(1.5*iqr.w_weight))
index=np.where(data['w_weight']>whole_weight_high)
data=data.drop(data.index[index])
data.reset_index()


# In[23]:


s_weight_high=(q3.shuck_weight+(1.5*iqr.shuck_weight))
index=np.where(data['shuck_weight']>s_weight_high)
data=data.drop(data.index[index])
data.reset_index()


# In[24]:


v_weight_high=(q3.v_weight+(1.5*iqr.v_weight))
index=np.where(data['v_weight']>v_weight_high)
data=data.drop(data.index[index])
data.reset_index()


# In[25]:


shell_weight_high=(q3.shell_weight+(1.5*iqr.shell_weight))
index=np.where(data['shell_weight']>shell_weight_high)
data=data.drop(data.index[index])
data.reset_index()


# In[26]:


plt.figure(figsize=(20,50))
graph=1
for column in x:
    if plotnumber<=50:
        ax=plt.subplot(10,5,graph)
        sns.boxplot(data=data[column])
        plt.xlabel(column,fontsize=20)
    graph+=1
plt.show()


# In[27]:


plt.figure(figsize=(20,15),facecolor="orange")
plotnumber=1
for column in data:
    if plotnumber<=9:
        ax=plt.subplot(3,3,plotnumber)
        sns.distplot(data[column])
        plt.xlabel(column,fontsize=10)
    plotnumber+=1
plt.show()


# In[28]:


plt.figure(figsize=(20,15), facecolor='red')
plotnumber=1
for column in data:
    if plotnumber<=9:
        ax=plt.subplot(3,3,plotnumber)
        plt.scatter(data[column],data["age"])
        plt.xlabel(column,fontsize=10)
        plt.ylabel('Age', fontsize=10)
    plotnumber+=1
plt.show()


# In[29]:


df_corr=data.corr().abs()
plt.figure(figsize=(15,10))
sns.heatmap(df_corr,annot=True, annot_kws={'size':16})
plt.show()


# In[30]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30)


# In[31]:


x_train.shape,x_test.shape,y_train.shape,y_test.shape


# In[32]:


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# In[33]:


clf=DecisionTreeClassifier()


# In[34]:


y


# In[35]:


clf.fit(x_train,y_train)


# In[36]:


clf.score(x_train,y_train)


# In[37]:


y_pred=clf.predict(x_test)


# In[38]:


accuracy_score(y_test,y_pred)


# In[39]:


#Let do some Hyperparameter tuning

grid_param={
    'criterion':['gini','entropy'],
    'max_depth':range(10,15),
    'min_samples_leaf':range(2,10),
    'min_samples_split': range(3,10),
    'max_leaf_nodes':range(2,4)
}


# In[40]:


grid_search=GridSearchCV(estimator=clf,param_grid=grid_param,cv=5,n_jobs=-1)


# In[41]:


grid_search.fit(x_train,y_train)


# In[42]:


best_parameters=grid_search.best_params_
print(best_parameters)


# In[43]:


clf=DecisionTreeClassifier(criterion='gini',max_leaf_nodes=20,min_samples_split=10,max_depth=60,min_samples_leaf=2)


# In[44]:


clf.fit(x_train,y_train)


# In[45]:


y_pred=clf.predict(x_test)


# In[46]:


accuracy_score(y_test,y_pred)


# In[ ]:




