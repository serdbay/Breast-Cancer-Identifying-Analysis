#!/usr/bin/env python
# coding: utf-8

# # Breast Cancer Identifying Analysis
# 
# ## Table of Contents
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#wrangling">Data Wrangling</a></li>
# <li><a href="#eda">Exploratory Data Analysis</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# </ul>

# <a id='intro'></a>
# ## Introduction
# 
# ### Dataset Description 

# In this project we will be analyzing Breast Cancer Data. This dataset is a comprehensive dataset that contains nearly all the PLCO study data available for breast cancer incidence and mortality analyses. For many women the trial documents multiple breast cancers, however, this file only has data on the earliest breast cancer diagnosed in the trial.

# Breast cancer is the most common cancer amongst women in the world. It accounts for 25% of all cancer cases, and affected over 2.1 Million people in 2015 alone. It starts when cells in the breast begin to grow out of control. These cells usually form tumors that can be seen via X-ray or felt as lumps in the breast area.
# 
# The key challenges against it’s detection is how to classify tumors into malignant (cancerous) or benign(non cancerous). We will try to complete the analysis of classifying these tumors using machine learning and the Breast Cancer Dataset.

# #### Objective:
# 
# * Understand the Dataset & cleanup (if required).
# 
# * Build classification models to predict whether the cancer type is Malignant or Benign.
# 
# * Also fine-tune the hyperparameters and compare the evaluation metrics of our model.

# <a id='wrangling'></a>
# ## Data Wrangling
# 
# ### General Properties

# In[2]:


#importing the data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


#loading our dataset
from sklearn.datasets import load_breast_cancer #this is our data
cancer=load_breast_cancer()


# In[4]:


cancer


# In[5]:


#checking the columns of our dataset 
cancer.keys()


# In[6]:


print(cancer['DESCR'])


# In[7]:


print(cancer['target'])


# In[8]:


print(cancer['target_names'])


# In[9]:


print(cancer['feature_names'])


# In[10]:


cancer['data'].shape


# In[11]:


#creating our dataframe
df_cancer=pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns=np.append(cancer['feature_names'], ['target'])) #np.c_ it concats your first array into the last dimension (axis) of your last array in the function.
df_cancer.head()


# In[12]:


df_cancer.tail()


# In[13]:


df_cancer.shape


# In[14]:


df_cancer.columns


# #### Data Cleaning

# In[15]:


df_cancer.isna().sum() #checking our null values. 


# We see that we don'a have any null values so there is no need to deal with the null values of our dataset. 

# <a id='eda'></a>
# ## Exploratory Data Analysis

# In[16]:


#visualizing the data
sns.pairplot(df_cancer, vars=['mean radius', 'mean texture', 'mean area', 'mean perimeter', 
       'mean smoothness'], hue='target')


# We can see the negative and positive relationships, distribution of our variables together in here with respect to target (hue) parameter. For example we can see positive relationship between mean radius, mean perimeter and mean area. 

# In[17]:


sns.countplot(df_cancer['target'])


# In[44]:


df_cancer.target.value_counts() #we can say in here that this is not an unbalanced dataset.


# In[45]:


sns.scatterplot(x='mean area', y='mean smoothness', hue='target', data=df_cancer)


# In[46]:


plt.figure(figsize=(23,13))
sns.heatmap(df_cancer.corr(), annot=True)


# In this heatmap we can see the relationships in our dataset with respect to their correlations. We know that the correlation coefficient gets bigger as it gets closer to 1 and gets smaller as it gets farther away from 1.

# #### Model Training

# In[20]:


X=df_cancer.drop(['target'], axis=1) #our input data
X


# In[21]:


y=df_cancer['target'] #our output data (target)
y


# In[22]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=5)


# In[23]:


X_train


# In[24]:


y_train


# In[25]:


#we will use SVM (support vector machine) for our classification, this algorithm can be use both for regression or classification.
from sklearn.svm import SVC #(support vector classifier)


# In[26]:


from sklearn.metrics import classification_report, confusion_matrix


# In[27]:


svc_model=SVC()


# In[28]:


svc_model.fit(X_train, y_train) #training the data (teaching the data)


# In[29]:


#Evaluating the model
y_predict=svc_model.predict(X_test) #prediction on the test data 


# In[30]:


y_predict


# In[31]:


cm=confusion_matrix(y_test, y_predict)
cm


# In[47]:


sns.heatmap(cm, annot=True, yticklabels=True, xticklabels=True) #we can see in here that we should improve the model because we have some unclassified values.


# In[48]:


#Improving the model
#feature scaling
min_train=X_train.min()
range_train=(X_train-min_train).max()
X_train_scaled=(X_train-min_train)/range_train #X-Xmin/Xmax-Xmin


# In[49]:


sns.scatterplot(x=X_train['mean area'], y=X_train['mean smoothness'], hue=y_train)


# In[50]:


sns.scatterplot(x=X_train_scaled['mean area'], y=X_train_scaled['mean smoothness'], hue=y_train)


# In[51]:


min_test=X_test.min()
range_test=(X_test-min_test).max()
X_test_scaled=(X_test-min_test)/range_test #X-Xmin/Xmax-Xmin


# In[52]:


svc_model.fit(X_train_scaled,y_train)


# In[53]:


y_predict=svc_model.predict(X_test_scaled)


# In[54]:


cm=confusion_matrix(y_test,y_predict)


# In[55]:


sns.heatmap(cm, annot=True) #now we can see that our model improved after scaling when we check the unclassified values. 


# In[56]:


print(classification_report(y_test,y_predict))


# Our scores looks good in our classification report in here. Let's improve our model parameters. 

# In[57]:


#Improving the parameters
param_grid={'C':[0.1, 1, 10, 100], 'gamma':[1, 0.1, 0.01, 0.001], 'kernel':['rbf']} #Radial Basis Function Kernel (rbf) or Gaussian kernel is a kernel function that is used in machine learning to find a non-linear classifier or a regression line.
param_grid #RBF (radial basis function kernel) is used in SVM because it helps the SVM to become non-linear rather than linear. RBF kernel function is similar to normal distribution.


# In[59]:


from sklearn.model_selection import GridSearchCV
grid=GridSearchCV(SVC(), param_grid, refit=True, verbose=4) #GridSearchCV is a technique to search through the best parameter values from the given set of the grid of parameters. It is basically a cross-validation method. And in here verbose shows us how many values that we wanted to display while searching for our grid.
grid


# In[60]:


grid.fit(X_train_scaled, y_train) #we are fitting our grid search CV in here. 


# In[61]:


grid.best_params_ #these are our best parameters


# In[62]:


grid_predictions=grid.predict(X_test_scaled)


# In[63]:


cm=confusion_matrix(y_test,grid_predictions)
cm


# In[64]:


sns.heatmap(cm, annot=True) #0 is the malignant and 1 is the benign classes as we know before and in here we see that our model is improved with the grid predictions only have 4 misclassified samples that is type 1 error.


# In[65]:


print(classification_report(y_test, grid_predictions)) #our model is improved as we can see in here. 


# <a id='conclusions'></a>
# ## Conclusions

# SVM is a supervised machine learning algorithm which can be used for classification or regression problems. It uses a technique called the kernel trick to transform your data and then based on these transformations it finds an optimal boundary between the possible outputs.
# 
# In this binary classification problem we use a machine learning technique Support Vector Machine (SVM).
# 
# We tried to classify breast cancer tumors into Malignant/Bening with approximately 97% accuracy. This technique can rapidly evaluate breast masses and classify them in an automated fashion.
# 
# Early breast cancer can dramatically save lives especially in the developing world.
# 
# The technique can be further improved by combining computer vision/machine learning techniques to directly classify cancer using tissue images.

# In[ ]:




