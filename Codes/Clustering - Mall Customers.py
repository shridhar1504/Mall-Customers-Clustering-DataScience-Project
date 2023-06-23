#!/usr/bin/env python
# coding: utf-8

# # Clustering Mall Customers

# **Importing the required libraries & packages**

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import ydata_profiling as pf
get_ipython().run_line_magic('matplotlib', 'inline')
from scipy.cluster.hierarchy import dendrogram,linkage
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import warnings
warnings.filterwarnings('ignore')
import pickle
from matplotlib.pylab import rcParams
rcParams['figure.figsize']=(20,10)


# **Changing The Default Working Directory Path & Reading the Dataset using Pandas Command**

# In[2]:


os.chdir('C:\\Users\\Shridhar\\OneDrive\\Desktop\\Top Mentor\\Batch 74 Day 18')
df=pd.read_csv('Mall_Customers.csv')


# **Automated Exploratory Data Analysis (EDA) with ydata_profiling(pandas_profiling)**

# In[3]:


pf.ProfileReport(df)


# **Checking the Null values of all the columns in the dataset.**

# In[4]:


df.isna().sum()


# **Assigning the independent variable since it is Clustering Model there's no dependent variable.**

# In[5]:


x=df.iloc[:,[3,4]].values


# **Finding the WCSS (Within Cluster Sum of Square) values using KMeans Clustering Model**

# In[6]:


wcss=[]
for i in range(2,11):
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
display(wcss)


# **Plotting the Line Graph with WCSS Values to get the exact ideal number of clusters to be created using KMeans Clustering Algorithm and saving the PNG file of the graph.**

# In[7]:


plt.plot(range(2,11),wcss)
plt.title('Elbow Method')
plt.xlabel('No. of Clusters')
plt.ylabel('WCSS')
plt.savefig('Elbow Method Graph.png')
plt.show()


# **Fitting the KMeans Clustering model with ideal number of clusters found from Elbow Method Graph and getting the dataset belonging to the Cluster.**

# In[8]:


kmeans=KMeans(n_clusters=5)
kmeans.fit(x)
y_kmeans=kmeans.labels_
display(y_kmeans)


# **Plotting the Scatter Plot Graph with the independent variable and the Cluster which it belongs and saving the PNG file.**

# In[9]:


plt.scatter(x[y_kmeans==0,0],x[y_kmeans==0,1],c='r',s=100,label='Cluster 1')
plt.scatter(x[y_kmeans==1,0],x[y_kmeans==1,1],c='b',s=100,label='Cluster 2')
plt.scatter(x[y_kmeans==2,0],x[y_kmeans==2,1],c='g',s=100,label='Cluster 3')
plt.scatter(x[y_kmeans==3,0],x[y_kmeans==3,1],c='m',s=100,label='Cluster 4')
plt.scatter(x[y_kmeans==4,0],x[y_kmeans==4,1],c='c',s=100,label='Cluster 5')
plt.title('KMeans Clustering Spread')
plt.savefig('Kmeans Clustering Spread Graph.png')
plt.legend()
plt.show()


# **Merging the Cluster Number and adding the sepearte column for it with Original Dataset and displaying the resulting dataset**

# In[10]:


result=pd.concat([df,pd.DataFrame(y_kmeans,columns=['Cluster Number'])],axis=1)
display(result)


# **Grouping By The Cluster Number to see the number of values in each Clusters**

# In[11]:


result.groupby('Cluster Number').size()


# **Grouping by the Cluster Number with respect to Annual Income and Spending Score to get the Minimum, Maximum values of Annual Income and Spending Score for each Clusters.**

# In[12]:


result.groupby('Cluster Number').agg({'Annual Income (k$)':[np.min,np.max],'Spending Score (1-100)':[np.min,np.max]})


# **Plotting the Dendrogram Graph using Ward Method to find out the exact ideal number of clusters to be created using Agglomerative Clustering Model with Ward Linkage and saving the PNG file**

# In[13]:


dend=dendrogram(linkage(x,method='ward'))
plt.title('Dendrogram - Ward')
plt.xlabel('Customers')
plt.ylabel('ED')
plt.savefig('Dendrogram Ward.png')
plt.show()


# **Fitting the Agglomerative Clustering model with ideal number of clusters found from the Dendrogram using Ward Method and predicting the dataset belonging to the Cluster.**

# In[14]:


hc=AgglomerativeClustering(n_clusters=5,linkage='ward')
y_hc=hc.fit_predict(x)


# **Plotting the Scatter Plot Graph with the independent variable and the Cluster which it belongs and saving the PNG file.**

# In[15]:


plt.scatter(x[y_hc==0,0],x[y_hc==0,1],c='r',s=100,label='Cluster 1')
plt.scatter(x[y_hc==1,0],x[y_hc==1,1],c='b',s=100,label='Cluster 2')
plt.scatter(x[y_hc==2,0],x[y_hc==2,1],c='g',s=100,label='Cluster 3')
plt.scatter(x[y_hc==3,0],x[y_hc==3,1],c='m',s=100,label='Cluster 4')
plt.scatter(x[y_hc==4,0],x[y_hc==4,1],c='c',s=100,label='Cluster 5')
plt.title('Hierarchical Clustering Spread - Ward')
plt.savefig('Hierarchical Clustering Spread - Ward.png')
plt.legend()
plt.show()


# **Merging the Cluster Number and adding the sepearte column for it with Original Dataset and displaying the resulting dataset**

# In[16]:


result_hc_ward=pd.concat([df,pd.DataFrame(y_hc,columns=['Cluster Number'])],axis=1)
display(result_hc_ward)


# **Grouping by the Cluster Number with respect to Annual Income and Spending Score to get the Minimum, Maximum values of Annual Income and Spending Score and the number of values in each Clusters.**

# In[17]:


result_hc_ward.groupby('Cluster Number').agg({'Annual Income (k$)':[np.min,np.max],'Spending Score (1-100)':[np.min,np.max,np.size]})


# **Plotting the Dendrogram Graph using Single Method to find out the exact ideal number of clusters to be created using Agglomerative Clustering Model with Single Linkage and saving the PNG file**

# In[18]:


dend=dendrogram(linkage(x,method='single'))
plt.title('Dendrogram - Single')
plt.xlabel('Customers')
plt.ylabel('ED')
plt.savefig('Dendrogram Single.png')
plt.show()


# **Fitting the Agglomerative Clustering model with ideal number of clusters found from the Dendrogram using Single Method and predicting the dataset belonging to the Cluster.**

# In[19]:


hc1=AgglomerativeClustering(n_clusters=7,linkage='single')
y_hc1=hc1.fit_predict(x)


# **Plotting the Scatter Plot Graph with the independent variable and the Cluster which it belongs and saving the PNG file.**

# In[20]:


plt.scatter(x[y_hc1==0,0],x[y_hc1==0,1],s=100,label='Cluster 1')
plt.scatter(x[y_hc1==1,0],x[y_hc1==1,1],s=100,label='Cluster 2')
plt.scatter(x[y_hc1==2,0],x[y_hc1==2,1],s=100,label='Cluster 3')
plt.scatter(x[y_hc1==3,0],x[y_hc1==3,1],s=100,label='Cluster 4')
plt.scatter(x[y_hc1==4,0],x[y_hc1==4,1],s=100,label='Cluster 5')
plt.scatter(x[y_hc1==5,0],x[y_hc1==5,1],s=100,label='Cluster 6')
plt.scatter(x[y_hc1==6,0],x[y_hc1==6,1],s=100,label='Cluster 7')
plt.title('Hierarchical Clustering Spread - Single')
plt.savefig('Hierarchical Clustering Spread - Single.png')
plt.legend()
plt.show()


# **Merging the Cluster Number and adding the sepearte column for it with Original Dataset and displaying the resulting dataset**

# In[21]:


result_hc_single=pd.concat([df,pd.DataFrame(y_hc1,columns=['Cluster Number'])],axis=1)
display(result_hc_single)


# **Grouping by the Cluster Number with respect to Annual Income and Spending Score to get the Minimum, Maximum values of Annual Income and Spending Score and the number of values in each Clusters.**

# In[22]:


result_hc_single.groupby('Cluster Number').agg({'Annual Income (k$)':[np.min,np.max],'Spending Score (1-100)':[np.min,np.max,np.size]})


# **Plotting the Dendrogram Graph using Complete Method to find out the exact ideal number of clusters to be created using Agglomerative Clustering Model with Complete Linkage and saving the PNG file**

# In[23]:


dend=dendrogram(linkage(x,method='complete'))
plt.title('Dendrogram - Complete')
plt.xlabel('Customers')
plt.ylabel('ED')
plt.savefig('Dendrogram Complete.png')
plt.show()


# **Fitting the Agglomerative Clustering model with ideal number of clusters found from the Dendrogram using Complete Method and predicting the dataset belonging to the Cluster.**

# In[24]:


hc2=AgglomerativeClustering(n_clusters=5,linkage='complete')
y_hc2=hc2.fit_predict(x)


# **Plotting the Scatter Plot Graph with the independent variable and the Cluster which it belongs and saving the PNG file.**

# In[25]:


plt.scatter(x[y_hc2==0,0],x[y_hc2==0,1],s=100,label='Cluster 1')
plt.scatter(x[y_hc2==1,0],x[y_hc2==1,1],s=100,label='Cluster 2')
plt.scatter(x[y_hc2==2,0],x[y_hc2==2,1],s=100,label='Cluster 3')
plt.scatter(x[y_hc2==3,0],x[y_hc2==3,1],s=100,label='Cluster 4')
plt.scatter(x[y_hc2==4,0],x[y_hc2==4,1],s=100,label='Cluster 5')
plt.title('Hierarchical Clustering Spread - Complete')
plt.savefig('Hierarchical Clustering Spread - Complete.png')
plt.legend()
plt.show()


# **Merging the Cluster Number and adding the sepearte column for it with Original Dataset and displaying the resulting dataset**

# In[26]:


result_hc_complete=pd.concat([df,pd.DataFrame(y_hc2,columns=['Cluster Number'])],axis=1)
display(result_hc_complete)


# **Grouping by the Cluster Number with respect to Annual Income and Spending Score to get the Minimum, Maximum values of Annual Income and Spending Score and the number of values in each Clusters.**

# In[27]:


result_hc_complete.groupby('Cluster Number').agg({'Annual Income (k$)':[np.min,np.max],'Spending Score (1-100)':[np.min,np.max,np.size]})


# **Plotting the Dendrogram Graph using Average Method to find out the exact ideal number of clusters to be created using Agglomerative Clustering Model with Average Linkage and saving the PNG file**

# In[28]:


dend=dendrogram(linkage(x,method='average'))
plt.title('Dendrogram - Average')
plt.xlabel('Customers')
plt.ylabel('ED')
plt.savefig('Dendrogram Average.png')
plt.show()


# **Fitting the Agglomerative Clustering model with ideal number of clusters found from the Dendrogram using Average Method and predicting the dataset belonging to the Cluster.**

# In[29]:


hc3=AgglomerativeClustering(n_clusters=7,linkage='average')
y_hc3=hc3.fit_predict(x)


# **Plotting the Scatter Plot Graph with the independent variable and the Cluster which it belongs and saving the PNG file.**

# In[30]:


plt.scatter(x[y_hc3==0,0],x[y_hc3==0,1],s=100,label='Cluster 1')
plt.scatter(x[y_hc3==1,0],x[y_hc3==1,1],s=100,label='Cluster 2')
plt.scatter(x[y_hc3==2,0],x[y_hc3==2,1],s=100,label='Cluster 3')
plt.scatter(x[y_hc3==3,0],x[y_hc3==3,1],s=100,label='Cluster 4')
plt.scatter(x[y_hc3==4,0],x[y_hc3==4,1],s=100,label='Cluster 5')
plt.scatter(x[y_hc3==5,0],x[y_hc3==5,1],s=100,label='Cluster 6')
plt.scatter(x[y_hc3==6,0],x[y_hc3==6,1],s=100,label='Cluster 7')
plt.title('Hierarchical Clustering Spread - Complete')
plt.savefig('Hierarchical Clustering Spread - Complete.png')
plt.legend()
plt.show()


# **Merging the Cluster Number and adding the sepearte column for it with Original Dataset and displaying the resulting dataset**

# In[31]:


result_hc_average=pd.concat([df,pd.DataFrame(y_hc3,columns=['Cluster Number'])],axis=1)
display(result_hc_average)


# **Grouping by the Cluster Number with respect to Annual Income and Spending Score to get the Minimum, Maximum values of Annual Income and Spending Score and the number of values in each Clusters.**

# In[32]:


result_hc_average.groupby('Cluster Number').agg({'Annual Income (k$)':[np.min,np.max],'Spending Score (1-100)':[np.min,np.max,np.size]})

