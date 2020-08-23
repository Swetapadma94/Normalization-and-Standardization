#!/usr/bin/env python
# coding: utf-8

# Why Transformation of features are required?
# 
# 1.Linear Regression----Gradient Descent----Global Minima
# 2.Algorithm----KNN,K-Means,Hierarchecal Clustering----Eucledian Distance
# 
# --> Every point has some vectors and Direction.
# 
# Questions
# 
# 1.Do we require transformation in ensemble techniqies(Random Forest,Decision Tree,XGBoost,AdaBoost)
# -->NO
# 2.Where we should use transformation?
# -->Linear,Logistics,KNN,K-Means,anything where eucledian distance or gradient descent  is used.
# 
# 3.Deep Learning Techniques.
# 
# > ANN---> Global Minima,Gradient Descent
# > CNN---> image(0-255) here we divide each pixel with 255.
# > RNN--->LSTN
# **** ANN and RNN both use Standardization and Normalization.

# ###Transfomation Features
# 1Normalization and Standardization.
# 2.scaling to minimum and maximum vaues.
# 3.scaling to median and Quantiles.
# 4.Gussian Transformation:
#   ..>Logarithm Transformation
#   ..>Reciprocal Transformation
#   ..>Square-root Transformation 
#   ..>Exponential Transformation
#   ..>Box Cox Transformation

# ## Standardization(centering variable at zero)
# We try to bring all  the variables or features into  a similar scale
# z=(x-x_mean)/sd
# > mean=0,stdev=1

# In[1]:


df=pd.read_csv(r'E:\Krish naik\kaggle dataset\Titanic\train.csv',encoding='latin1',usecols=['Pclass','Age','Fare','Survived'])


# In[2]:


df.head()


# In[3]:


df.isnull().mean()


# In[4]:


df['Age'].fillna(df.Age.median(),inplace=True)


# In[5]:


df.isnull().sum()


# In[6]:


# Standardization: We use standardscaler from sklearn library
from sklearn.preprocessing import StandardScaler


# In[8]:


scaler=StandardScaler()
# fit vs fit_transform(to train model-fit,to apply on data-fit_transform)
df_scaled=scaler.fit_transform(df)


# In[9]:


df_scaled


# In[10]:


scaled=pd.DataFrame(df_scaled)


# In[11]:


# Transformation happened based on columns.
scaled


# In[13]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[15]:


plt.hist(df_scaled[:,1],bins=20)


# In[16]:


plt.hist(df_scaled[:,2],bins=20)


# In[17]:


plt.hist(df_scaled[:,3],bins=20)


# ## Min Max Scaling-works well with CNN
# Min Max scaling scales the values btween  and 1
# (x-x.min)/(x.max-x.min)

# In[18]:


from sklearn.preprocessing import MinMaxScaler


# In[19]:


min_max=MinMaxScaler()


# In[25]:


df_minmax=pd.DataFrame(min_max.fit_transform(df),columns=df.columns)


# In[27]:


df_minmax.head()


# In[29]:


plt.hist(df_minmax['Pclass'],bins=20)


# In[30]:


plt.hist(df_minmax['Age'],bins=20)


# ### Roboust Scaler
# It is used to scale the feature to median and quantiles.
# Scaling using median and quantiles consits of substracting the median to all the observations,and then dividing by the interquantile difference
# IQR=75th Quantile- 25th Quantile
# (x-x.median)/IQR
# Range-0 to 1
# 0,1,2,3,4,5,6,7,8,9,10
# 9-90 percentile--- 90% of all values in this group is less than 
# 4-40

# In[31]:


from sklearn.preprocessing import RobustScaler


# In[32]:


scaler=RobustScaler()


# In[35]:


df_robust_scaler=pd.DataFrame(scaler.fit_transform(df),columns=df.columns)


# In[37]:


df_robust_scaler


# In[39]:


plt.hist(df_robust_scaler['Age'],bins=20)


# In[40]:


plt.hist(df_robust_scaler['Fare'],bins=20)


# ## Gussian Transformation
# Some ML algorithm like linear and logistic assume that features are normally distributed.
# Advantages
# > Accuracy
# > Performance
# Gussian Distribution is same as Normal Distribution
# Basically we apply it on skewed data.
# ..>Logarithm Transformation
#   ..>Reciprocal Transformation
#   ..>Square-root Transformation 
#   ..>Exponential Transformation
#   ..>Box Cox Transformation

# In[41]:



df=pd.read_csv(r'E:\Krish naik\kaggle dataset\Titanic\train.csv',encoding='latin1',usecols=['Age','Fare','Survived'])


# In[42]:


df.head()


# In[45]:


df['Age']=df['Age'].fillna(df.Age.median())


# In[46]:


df.isna().sum()


# In[49]:


#If we want to check weather feature is gussian or normal distributed or not,we use Q-Q plot.
import scipy.stats as stat
import pylab
def plot_data(df,variable):
    plt.figure(figsize=(10,6))
    plt.subplot(1,2,1)
    df[variable].hist()
    plt.subplot(1,2,2)
    stat.probplot(df[variable],dist='norm',plot=pylab)
    


# In[50]:



plot_data(df,'Age')


# In[51]:


plot_data(df,'Fare')


# In[52]:


##-->1.Logarithm Transformation
# It works well for skewed data.
df['Age_log']=np.log(df['Age'])
plot_data(df,'Age_log')


# In[55]:


##-->2.Reciprocal  Transformation
df['Age-reciprocal']=1/df.Age
plot_data(df,'Age-reciprocal')


# In[57]:


#..>Square-root Transformation
df['Age_sqr']=df['Age']**(1/2)
plot_data(df,'Age_sqr')


# In[59]:


##..>Exponential Transformation
df['Age_xpon']=df['Age']**(1/1.2)
plot_data(df,'Age_xpon')


# ####..>BoxCox Transformation
# T(Y)=(Y-exp(lambda)-1)/lambda
# 

# In[62]:


df['Age_boxcox'],parameters=stat.boxcox(df['Age'])


# In[63]:


df['Age_boxcox'],parameters


# In[64]:


print(parameters)


# In[65]:


plot_data(df,'Age_boxcox')


# In[67]:


# IF we have 0 values in feature then we can't apply log so inthat case apply log1p.
df['Fare_log']=np.log1p(df['Fare'])
plot_data(df,'Fare_log')


# In[69]:


# +1 for negative values.
df['Fare_boxcox'],parameters=stat.boxcox(df['Fare']+1)
plot_data(df,'Fare_boxcox')


# In[ ]:




