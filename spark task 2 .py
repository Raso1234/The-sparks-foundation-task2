#!/usr/bin/env python
# coding: utf-8

# # The Spark foundation - Data science and business analysis
# 
# Task 2 - Prediction Using unsupervised machine learning
# 
# Task by - Rahul Kumar Maurya

# # Step 1 - Importing Required Libraries

# In[3]:


import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.metrics import silhouette_score
import warnings as wg
wg.filterwarnings('ignore')


# In[4]:


df=pd.read_csv(r"C:\Users\rishabh\Downloads\Iris.csv")
 #printing head of the data


# In[5]:


df.head()


# In[ ]:





# # Step 2 - Reading in the data from the source

# # Step 3 - Finding some properties of the data

# In[10]:


data.shape


# In[30]:


#checking for null values
data.isnull().sum()


# In[12]:


#checking the shape of data
data.info()


# In[31]:


#finding statistical properties of the data
data.describe()


# # Step 4 - Preparation of the data

# In[32]:


iris = pd.DataFrame(data)
#dropping the column of Id and species
iris_df = iris.drop(columns = ["Species","Id"])
#displaying the rearranged data
print(iris_df.head())
print(iris_df.tail())


# # Step 5 - Cheking for ouliers and removing them

# In[33]:


#checking for outliers by boxplot
plt.rcParams["figure.figsize"]=[10,8]
iris_df.plot(kind="box")
plt.show()


# In[34]:


#removing the ouliers using IQR method
Q1= iris_df.quantile(0.25)
Q3 = iris_df.quantile(0.75)
IQR = Q3-Q1
iris_df= iris_df[~((iris_df<(Q1-1.5*IQR)) | (iris_df>(Q3+1.5*IQR))).any(axis=1)]


# In[35]:


#ploting the boxplot after removing the ouliers
plt.rcParams["figure.figsize"]=(10,8)
iris_df.plot(kind="box")
plt.show()


# # Step 6 - K-Mean clustering  

# In[ ]:


Optimal value k using Elbow plot


# In[36]:


#data arrangment
x = iris_df.iloc[:,[0,1,2,3]].values


# In[37]:


from sklearn.cluster import KMeans
#create several cluster combinations and observe the wcss(within cluster sum of squares)
wcss=[]
K = range(1,11)
for i in K:
    kmeans=KMeans(n_clusters=i,init='k-means++', max_iter=300,n_init=10,random_state=0)
    kmeans=kmeans.fit(x)
    wcss.append(kmeans.inertia_)
wcss


# In[38]:


plt.plot(K,wcss,"go--")
plt.title("The Elbow Method for optimal K")
plt.xlabel("Number of clusters")
plt.ylabel("with clusters Sum of Square(wcss)")
plt.annotate("Elbow",xytext=(4,200),xy=(2.4,120),arrowprops={"facecolor":"blue"})
plt.grid()
plt.show()


# In[ ]:


optimal value of K using silhouette plot


# In[39]:


#initialise kmeans
kmeans = [KMeans(n_clusters=k,random_state=42).fit(x) for k in range(2,11)]
s=[silhouette_score(x, model.labels_)
   for model in kmeans[1:]]


# In[46]:


#plotting silhoutte visualiser
from yellowbrick.cluster import SilhouetteVisualizer

fig,ax = plt.subplots(2, 2, figsize=(15,8))
for i in [2, 3, 4, 5]:
    
    #create kmeans instance for different number of clusters
    
    km= KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=100, random_state=42)
    km_labels=km.fit_predict(x)
    q, mod = divmod(i,2)
    
    visualizer = SilhouetteVisualizer(km, colors='yellowbrick', ax=ax[q-1][mod])
    visualizer.fit(x)
    
    


# In[17]:


#plotting the silhouette score
plt.plot(range(3,11),s)
plt.xlabel("values of k")
plt.ylabel("Silhouette Score")
plt.title("Silhouette analysis for optimal k")
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




