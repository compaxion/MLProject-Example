#!/usr/bin/env python
# coding: utf-8

# # MachineLearning Project Example:

# ## Loading Data :

# In[1]:


import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/delaney_solubility_with_descriptors.csv')
df


# ## Data Preparation

# In[3]:


y = df['logS']
y


# In[5]:


x = df.drop('logS', axis = 1)
x


# ### Data Splitting

# In[7]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 100)


# In[9]:


x_train #0.8 of actual data as train set


# In[11]:


x_test #0.2 of actual data as testing set


# ## Model Building

# ### Linear Regression

# Training the Model:

# In[13]:


from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x_train, y_train)


# Applying the Model to Make a Prediction:

# In[15]:


y_lr_train_pred = lr.predict(x_train)
y_lr_test_pred = lr.predict(x_test)


# In[19]:


y_lr_train_pred


# In[21]:


y_lr_test_pred


# Evaluate model performance:

# In[27]:


from sklearn.metrics import mean_squared_error, r2_score

lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
lr_train_r2 = r2_score(y_train, y_lr_train_pred)

lr_test_mse = mean_squared_error(y_test, y_lr_test_pred)
lr_test_r2 = r2_score(y_test, y_lr_test_pred)


# In[47]:


lr_results = pd.DataFrame(['Linear Regression', lr_train_mse, lr_train_r2, lr_test_mse, lr_test_r2]).transpose()
lr_results.columns = ['Method','Training MSE','Training R2','Testing MSE','Testing R2']
lr_results


# ### Random Forest

# Training the Model:

# In[49]:


from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(max_depth = 2, random_state = 100)
rf.fit(x_train, y_train)


# Applying Model to Make Prediction:

# In[53]:


y_rf_train_pred = rf.predict(x_train)
y_rf_test_pred = rf.predict(x_test)


# Evaluate Model Performance:

# In[55]:


from sklearn.metrics import mean_squared_error, r2_score

rf_train_mse = mean_squared_error(y_train, y_rf_train_pred)
rf_train_r2 = r2_score(y_train, y_rf_train_pred)

rf_test_mse = mean_squared_error(y_test, y_rf_test_pred)
rf_test_r2 = r2_score(y_test, y_rf_test_pred)


# In[57]:


rf_results = pd.DataFrame(['Random Forest', rf_train_mse, rf_train_r2, rf_test_mse, rf_test_r2]).transpose()
rf_results.columns = ['Method','Training MSE','Training R2','Testing MSE','Testing R2']
rf_results


# ### Model Comparison

# In[61]:


df_models = pd.concat([lr_results, rf_results], axis = 0)
df_models


# In[65]:


df_models.reset_index(drop = True)


# # Data Visualization of Prediction Results

# In[77]:


import matplotlib.pyplot as plt
import numpy as np


plt.figure(figsize = (5,5))
plt.scatter(x = y_train, y = y_lr_train_pred, c = '#7CAE00', alpha = 0.2)

z = np.polyfit(y_train, y_lr_train_pred, 1)
p = np.poly1d(z)

plt.plot(y_train, p(y_train), '#F8766D')
plt.ylabel('Predict LogS')
plt.xlabel('Experimental LogS')
plt.plot()

