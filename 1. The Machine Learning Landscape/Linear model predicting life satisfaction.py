#!/usr/bin/env python
# coding: utf-8

# In[77]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model
import os


# In[78]:


os.getcwd()


# In[79]:


def prepare_country_stats(oecd_bli, gdp_per_capita):
    oecd_bli = oecd_bli[oecd_bli["INEQUALITY"]=="TOT"]
    oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")
    gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
    gdp_per_capita.set_index("Country", inplace=True)
    full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita,
                                  left_index=True, right_index=True)
    full_country_stats.sort_values(by="GDP per capita", inplace=True)
    remove_indices = [0, 1, 6, 8, 33, 34, 35]
    keep_indices = list(set(range(36)) - set(remove_indices))
    return full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[keep_indices]


# In[80]:


import os
datapath = os.path.join("datasets", "lifesat", "")


# In[81]:


# Download the data
import urllib.request
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
os.makedirs(datapath, exist_ok=True)
for filename in ("oecd_bli_2015.csv", "gdp_per_capita.csv"):
    print("Downloading", filename)
    url = DOWNLOAD_ROOT + "datasets/lifesat/" + filename
    urllib.request.urlretrieve(url, datapath + filename)


# In[82]:


# Load the data
oecd_bli = pd.read_csv(datapath + "oecd_bli_2015.csv", thousands=',')
gdp_per_capita = pd.read_csv(datapath + "gdp_per_capita.csv", thousands=',', delimiter='\t', encoding='latin1', na_values='na')


# In[95]:


gdp_per_capita.head()


# In[83]:


# Prepare the data
country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)
X = np.c_[country_stats["GDP per capita"]]
y = np.c_[country_stats["Life satisfaction"]]


# In[84]:


# Visualize the data
country_stats.plot(kind='scatter', x='GDP per capita', y='Life satisfaction')
plt.show()


# In[85]:


# Select a linear model
model = sklearn.linear_model.LinearRegression()


# In[87]:


# Train linear regression model
model.fit(X,y)


# In[88]:


# Make a prediction for Cyprus
X_new = [[22587]] # Cyprus's GDP per capita
print(model.predict(X_new)) # outputs [[5.96242338]]


# In[91]:


# Select k-Nearest Neighbors model
import sklearn.neighbors
model2 = sklearn.neighbors.KNeighborsRegressor(n_neighbors=3)


# In[92]:


# Train model
model2.fit(X,y)


# In[93]:


# Make a prediction for Cyprus
X_new = [[22587]] # Cyprus's GDP per capita
print(model2.predict(X_new)) # outputs [[5.96242338]]

