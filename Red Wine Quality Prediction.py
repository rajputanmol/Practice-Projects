#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


import warnings
warnings.filterwarnings('ignore')


# In[ ]:


data=pd.read_csv('https://raw.githubusercontent.com/dsrscientist/DSData/master/winequality-red.csv')
data.tail()


# In[ ]:


data.shape


# In[ ]:


data.describe()


# In[ ]:


data.isna().sum().sum()


# In[ ]:


#Since, there is no null values present, so let's plot heat map to check collinearity

df_corr=data.corr().abs()
plt.figure(figsize=(22,16))
sns.heatmap(df_corr,annot=True, annot_kws={'size':12})
plt.show()


# In[ ]:


#Everything seems good now, let's proceed to split features and label

x=data.drop(columns='quality', axis=1)
y=data['quality']


# In[ ]:


x


# In[ ]:


y


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)


# In[ ]:


x_train.shape,x_test.shape,y_train.shape,y_test.shape


# In[ ]:


#Let's visualize the tree and see how it looks like without any preprocessing
clf=DecisionTreeClassifier()
clf.fit(x_train,y_train)


# In[ ]:


feature_name=list(x.columns)
class_name=list(y_train.unique())


# In[ ]:


conda install graphviz


# In[ ]:


pip install graphviz


# In[ ]:


import graphviz
from IPython.display import Image
import pydotplus
from sklearn.tree import export_graphviz
from sklearn import tree

dot_data=export_graphviz(clf,feature_names = feature_name, rounded=True, filled=True)

#Draw Graph
graph=pydotplus.graph_from_dot_data(dot_data)
graph.write_png('mytree.png')
#Show Graph
Image(graph.create_png())


# In[ ]:


clf.score(x_train,y_train)


# In[ ]:


y_pred=clf.predict(x_test)


# In[ ]:


accuracy_score(y_test,y_pred)


# In[ ]:


#Let's do some Hyperparameter tuning using GridSearchCV algorithm
grid_param={
    'criterion':['gini','entropy'],
    'max_depth':range(10,20),
    'min_samples_leaf':range(2,15),
    'min_samples_split': range(2,10),
    'max_leaf_nodes':range(2,5)
}


# In[ ]:


grid_search=GridSearchCV(estimator=clf,param_grid=grid_param,cv=5,n_jobs=-1)


# In[ ]:


grid_search.fit(x_train,y_train)


# In[ ]:


best_parameters=grid_search.best_params_
print(best_parameters)


# In[ ]:


clf=DecisionTreeClassifier(criterion='gini',max_leaf_nodes=30,min_samples_split=20,max_depth=5,min_samples_leaf=3)


# In[ ]:


clf.fit(x_train,y_train)


# In[ ]:


y_pred=clf.predict(x_test)


# In[ ]:


accuracy_score(y_test,y_pred)


# In[ ]:




