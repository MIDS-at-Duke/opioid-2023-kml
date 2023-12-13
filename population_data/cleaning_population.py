#!/usr/bin/env python
# coding: utf-8

# Source of the population data: https://seer.cancer.gov/popdata/singleages.html

# In[1]:


import pandas as pd

pd.set_option("mode.copy_on_write", True)


# In[2]:


# Define the column widths for each field in the fixed-length data
column_widths = [4, 2, 2, 3, 2, 1, 1, 1, 2, 8]
# Define the column names
column_names = [
    "Year",
    "State",
    "State_FIPS",
    "County_FIPS",
    "Registry",
    "Race",
    "Origin",
    "Sex",
    "Age",
    "Population" "County",
]

# Read the fixed-length data file with column names
data = pd.read_fwf(
    "population_death/us.1969_2020.19ages.adjusted.txt",
    widths=column_widths,
    names=column_names,
)

# Display the data
print(data)


# In[3]:


# Tried to use dask to read the data, but it did not get much faster
# import dask.dataframe as dd
# import pandas as pd

# # Define the column widths for each field in the fixed-length data
# column_widths = [4, 2, 2, 3, 2, 1, 1, 1, 2, 8]
# # Define the column names
# column_names = ['Year', 'State', 'State_FIPS', 'County_FIPS', 'Registry', 'Race', 'Origin', 'Sex', 'Age', 'Population']

# ddf = dd.read_fwf('us.1969_2020.19ages.adjusted.txt', widths=column_widths, names=column_names)
# data = ddf.compute()


# In[4]:


data


# In[5]:


count_combinations = (
    data.groupby(["State_FIPS", "County_FIPS"]).size().reset_index(name="Count")
)
print(count_combinations)


# In[6]:


year_count = data.groupby(["Year"]).size().reset_index(name="Count")
year_count


# In[7]:


selected_years = data[(data["Year"] >= 2002) & (data["Year"] <= 2016)]
selected_years


# In[8]:


selected_years["FIPS"] = selected_years["State_FIPS"].astype(str).str.zfill(
    2
) + selected_years["County_FIPS"].astype(str).str.zfill(3)


# In[9]:


collapsed_data = (
    selected_years.groupby(["FIPS", "Year", "State", "State_FIPS", "County_FIPS"])
    .sum()
    .reset_index()
)
collapsed_data = collapsed_data[
    ["FIPS", "Year", "State", "State_FIPS", "County_FIPS", "Population"]
]

collapsed_data


# In[10]:


melted_data = collapsed_data.pivot_table(
    index=["FIPS", "State", "State_FIPS", "County_FIPS"],
    columns="Year",
    values="Population",
).reset_index()
melted_data.columns.name = None

melted_data


# In[11]:


rows_with_nan = melted_data[melted_data.isnull().any(axis=1)]
print(rows_with_nan["FIPS"])


# * There are 9 counties with missing values during the year ranging 2002-2016, so we drop these counties
# 02105 02195 02198 02201 02230 02232 02275 02280 99999

# In[12]:


melted_data.dropna(inplace=True)
melted_data


# * There are 3143 counties in the US now, and the data has 3137 rows, covered almost all the counties

# In[13]:


melted_data.to_csv("population.csv", index=False)
