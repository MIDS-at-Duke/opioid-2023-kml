# %% [markdown]
# Step 1: Load the data

# %%
import pandas as pd
import numpy as np


pd.set_option("mode.copy_on_write", True)

# %%
death = pd.read_csv("state_text_files/drug_deaths.txt", delimiter="\t")

# %%
population = pd.read_csv("population_data/population.csv")

# %%
population

# %% [markdown]
# Step 2: Clean the County Code

# %%
death.head(10)

# %% [markdown]
# * We will merge death's Couty Code with populaiton's FIPS. Couty code is now a float so we need to convert it to int.

# %%
death["County Code"] = pd.to_numeric(
    death["County Code"], errors="coerce"
)  # Convert to numeric
death["County Code"] = death["County Code"].astype(int)

# %%
death.head(10)

# %% [markdown]
# Step 3: Subset the population data based on states and years
# * We can get the state code from the County Code. In this way, we double check if we get the correct state. Also we can subset population df with only the relevant states, and we can increase our data processing time by subsetting it.

# %%
import math


death["State Code"] = death["County Code"].apply(lambda x: float(math.floor(x / 1000)))


unique_state_code = death["State Code"].unique()


print(unique_state_code)

# %% [markdown]
# * There are 12 unique state code from the death dataset: [17. 39. 12. 47. 48. 53. 54. 20. 41. 31. 56.]. So we selected the data correctly
# * Next step we subset the popultion data based on the state code.

# %%
subset_population = population[population["State_FIPS"].isin(unique_state_code)]

# %%
print(death["Year"].min())
print(death["Year"].max())

# %%
subset_population = subset_population.drop(["2002", "2016"], axis=1)

# %%
subset_population

# %% [markdown]
# Step 4: Check the valid counties number

# %%
print(subset_population["FIPS"].nunique())
print(death["County Code"].nunique())

# %% [markdown]
# death_unique_county = death["County Code"].unique()
# population_unique_county = death[]

# %% [markdown]
# #### Lets merge the two datasets

# %%
subset_population.head(10)

# %%
death.head(10)

# %%
death = death.rename(columns={"County Code": "FIPS"})

# %%
death.head(10)

# %% [markdown]
# > I will get the 'Year' Column to equal the column of that matching Year and then insert that value into a column called 'Population' so to make the dataframe more digestable.

# %%
subset_population.columns

# %%
columns_to_melt = [
    "2003",
    "2004",
    "2005",
    "2006",
    "2007",
    "2008",
    "2009",
    "2010",
    "2011",
    "2012",
    "2013",
    "2014",
    "2015",
]  # update the list base on your year
melted_df = pd.melt(
    subset_population,
    id_vars=["FIPS", "State", "County_FIPS"],
    value_vars=columns_to_melt,
    var_name="Year",
    value_name="Population",
)
melted_df["Year"] = melted_df["Year"].astype(int)
melted_df

# %% [markdown]
#
# unique_years = merged_df["Year"].unique()
#
#
# merged_df["Population"] = merged_df.apply(
#     lambda row: row[str(int(row["Year"]))]
#     if pd.notna(row["Year"]) and int(row["Year"]) in unique_years
#     else None,
#     axis=1,
# )
#
# merged_df.sample(10)

# %% [markdown]
# > Check to make sure it kept every value: 2983 rows

# %%
merged_df = pd.merge(death, melted_df, on=["Year", "FIPS"], how="outer")
merged_df

# %%
merged_df.shape

# %%
adams_county_oh = merged_df[merged_df["County"] == "Adams County, OH"]
adams_county_oh

# %%
adams_FIPS = merged_df[merged_df["FIPS"] == 39001]
adams_FIPS

# %% [markdown]
# > It looks like it worked properly I will drop the year columns now

# %%
# Texas has 254 counties, I am making sure no filtering as removed all counties yet.
len(merged_df[merged_df["State"] == "TX"]["County_FIPS"].unique())

# %%
len(merged_df[merged_df["State"] == "FL"]["County_FIPS"].unique())

# %%
len(merged_df[merged_df["State"] == "WA"]["County_FIPS"].unique())

# %% [markdown]
# > All counties are registerd in our dataframe

# %%
# Just get counties that have NAN Deaths

no_drug_deaths = merged_df[merged_df["Deaths"].isna()]
no_drug_deaths.shape

# %%
no_drug_deaths.head(10)

# %%
no_drug_deaths[no_drug_deaths["FIPS"] == 12003]

# %%
grouped_by_FIPS = no_drug_deaths.groupby("FIPS")

# Filter FIPS numbers where all years have NaN for deaths
fips_with_12_nan = grouped_by_FIPS.filter(
    lambda group: group["Deaths"].isna().sum() == 12
)["FIPS"].unique()

# Display the FIPS numbers
print(len(fips_with_12_nan))

# %% [markdown]
# > 47 Counties have NO DEATH DATA across ALL years

# %%
all_yrs_missing = no_drug_deaths[no_drug_deaths["FIPS"].isin(fips_with_12_nan)]
all_yrs_missing = all_yrs_missing.groupby("FIPS")
concatenated_df = pd.concat([group_df for _, group_df in all_yrs_missing])

concatenated_df.head(17)

# %%
state_population_stats = (
    concatenated_df.groupby("State")["Population"]
    .agg(["min", "max", "mean", "median"])
    .reset_index()
)

# Rename the columns for clarity
state_population_stats.columns = [
    "State",
    "MinPopulation",
    "MaxPopulation",
    "MeanPopulation",
    "MedianPopulation",
]

# Print or display the resulting DataFrame
state_population_stats

# %%
state_population_stats_ALLDATA = (
    merged_df.groupby("State")["Population"]
    .agg(["min", "max", "mean", "median"])
    .reset_index()
)

# Rename the columns for clarity
state_population_stats_ALLDATA.columns = [
    "State",
    "MinPopulation",
    "MaxPopulation",
    "MeanPopulation",
    "MedianPopulation",
]

# Print or display the resulting DataFrame
state_population_stats_ALLDATA

# %%
compare_df = pd.DataFrame(
    {
        "State": state_population_stats["State"],
        "Max For Missing Counties": state_population_stats["MaxPopulation"],
        "Mean For Missing Counties ": state_population_stats["MeanPopulation"],
        "Median For All Counties ": state_population_stats_ALLDATA["MedianPopulation"],
        "Mean For All Counties ": state_population_stats_ALLDATA["MeanPopulation"],
    }
)

# Print or display the resulting DataFrame
compare_df

# %%
data = {
    "State": ["FL", "WV", "TX", "WA", "OR"],
    "Max For Missing Counties": [112354.0, 56841.0, 163331.0, 59970.0, 66767.0],
    "Mean For Missing Counties": [
        63225.458333,
        41382.25,
        58547.370370,
        57946.5,
        49506.305556,
    ],
    "Median For All Counties": [100204.0, 59785.0, 31032.0, 18223.5, 57926.0],
    "Mean For All Counties": [
        278765.39265,
        170142.443787,
        65923.953846,
        97329.838583,
        130962.466783,
    ],
}

compare_df2 = pd.DataFrame(data)

# Print or display the resulting DataFrame
compare_df2

# %% [markdown]
# I will filter by median population for all counties for each year to scope the analysis for larger counties in the state.

# %%
print(merged_df["Year"].unique())
print(merged_df["State"].unique())

# %%
states_to_keep = ["FL", "TX", "WA", "OR", "KS", "WV"]

# Filter DataFrame to include only specified states
filtered_df = merged_df[merged_df["State"].isin(states_to_keep)].copy()

# Print or display the resulting DataFrame
filtered_df

# %%
import pandas as pd

filtered_dfs_by_state = []

for state in filtered_df["State"].unique():
    state_df = filtered_df[filtered_df["State"] == state].copy()

    for year in state_df["Year"].unique():
        median_population = state_df.loc[
            state_df["Year"] == year, "Population"
        ].median()

        state_df_filtered = state_df[
            (state_df["Year"] == year) & (state_df["Population"] >= median_population)
        ]

        filtered_dfs_by_state.append(state_df_filtered)


filtered_dfs_by_state = pd.concat(filtered_dfs_by_state)

filtered_dfs_by_state

# %%
grouped_by_FIPS2 = filtered_dfs_by_state.groupby("FIPS")

# Filter FIPS numbers where all years have NaN for deaths
fips_with_12_nan2 = grouped_by_FIPS2.filter(
    lambda group: group["Deaths"].isna().sum() == 12
)["FIPS"].unique()

# Display the FIPS numbers
print(len(fips_with_12_nan2))

#

# %% [markdown]
# > Still 11 counties which have above the median population and have NO data for any year.
#
# > Just dropping for now... will revisit later

# %%
fips_with_12_nan2 = grouped_by_FIPS2.filter(
    lambda group: group["Deaths"].isna().sum() == 12
)["FIPS"].unique()

# Remove entries in filtered_dfs_by_state where FIPS is in fips_with_12_nan2
filtered_dfs_by_state = filtered_dfs_by_state[
    ~filtered_dfs_by_state["FIPS"].isin(fips_with_12_nan2)
]

# Display the result DataFrame
filtered_dfs_by_state

# %%
missing_Data = filtered_dfs_by_state[filtered_dfs_by_state["Deaths"].isna()]
len(missing_Data)

# %% [markdown]
# > Still 1944 missing drug deaths across all the counties

# %%
filtered_dfs_by_state[filtered_dfs_by_state["FIPS"] == 12001]

# %%
missing_data_by_year_county_state = (
    filtered_dfs_by_state[filtered_dfs_by_state["Deaths"].isna()]
    .groupby(["Year", "State"])
    .size()
    .reset_index(name="MissingCount")
)
missing_data_by_year_county_state

# %%
# Define the years for each group
post_years = [2012, 2013, 2014]
pre_years = [2011, 2010, 2009]

# Filter the DataFrame for the specified years and states
WA_OR = filtered_dfs_by_state[
    (
        (filtered_dfs_by_state["Year"].isin(pre_years))
        & (filtered_dfs_by_state["State"].isin(["WA", "OR"]))
    )
    | (
        (filtered_dfs_by_state["Year"].isin(post_years))
        & (filtered_dfs_by_state["State"].isin(["WA", "OR"]))
    )
]

# Display the result DataFrame
WA_OR

# %%
missing_data_WA_OR = (
    WA_OR[WA_OR["Deaths"].isna()]
    .groupby(["Year", "State"])
    .size()
    .reset_index(name="MissingCount")
)
print(missing_data_WA_OR["MissingCount"].sum())
missing_data_WA_OR

# %%
WA_OR_cleaned = WA_OR.dropna(subset=["Deaths"])

# Display the sum of missing counts and the DataFrame after dropping NaN values
print(f"Sum of missing counts: {missing_data_WA_OR['MissingCount'].sum()}")
WA_OR_cleaned

# %%
post = [2007, 2008, 2009]
pre = [2006, 2005, 2004]

# Filter the DataFrame for the specified years and states
TX_KS = filtered_dfs_by_state[
    (
        (filtered_dfs_by_state["Year"].isin(pre))
        & (filtered_dfs_by_state["State"].isin(["TX", "KS"]))
    )
    | (
        (filtered_dfs_by_state["Year"].isin(post))
        & (filtered_dfs_by_state["State"].isin(["TX", "KS"]))
    )
]

# Display the result DataFrame
TX_KS

# %%
missing_data_TX_KS = (
    TX_KS[TX_KS["Deaths"].isna()]
    .groupby(["Year", "State"])
    .size()
    .reset_index(name="MissingCount")
)
print(missing_data_TX_KS["MissingCount"].sum())
missing_data_TX_KS

# %%
TX_KS_cleaned = TX_KS.dropna(subset=["Deaths"])

# Display the sum of missing counts and the DataFrame after dropping NaN values
print(f"Sum of missing counts: {missing_data_TX_KS['MissingCount'].sum()}")
TX_KS_cleaned

# %%
post_fl = [2010, 2011, 2012]
pre_fl = [2009, 2008, 2007]

# Filter the DataFrame for the specified years and states
FL_WV = filtered_dfs_by_state[
    (
        (filtered_dfs_by_state["Year"].isin(pre_fl))
        & (filtered_dfs_by_state["State"].isin(["FL", "WV"]))
    )
    | (
        (filtered_dfs_by_state["Year"].isin(post_fl))
        & (filtered_dfs_by_state["State"].isin(["FL", "WV"]))
    )
]

# Display the result DataFrame
FL_WV

# %%
missing_data_FL_WV = (
    FL_WV[FL_WV["Deaths"].isna()]
    .groupby(["Year", "State"])
    .size()
    .reset_index(name="MissingCount")
)
print(missing_data_FL_WV["MissingCount"].sum())
missing_data_FL_WV

# %% [markdown]
# > There are very FEW missing data just going to drop for NOW

# %%
FL_WV_cleaned = FL_WV.dropna(subset=["Deaths"])

# Display the sum of missing counts and the DataFrame after dropping NaN values
print(f"Sum of missing counts: {missing_data_FL_WV['MissingCount'].sum()}")
FL_WV_cleaned

# %%


# %% [markdown]
# > Each comparison has 200-300 observations for all years needed for each comparison. This is good. We will need to look into normalizing/filling in missing data.

# %%
WA_OR_cleaned["DeathRate"] = WA_OR_cleaned["Deaths"] / WA_OR_cleaned["Population"]

# %%
wa = WA_OR_cleaned[WA_OR_cleaned["State"] == "WA"]
pre_wa_graph = wa[wa["Year"].isin(pre_years)]
post_wa_graph = wa[wa["Year"].isin(post_years)]

# plot pre_fl and post_fl in one graph
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))

sns.lineplot(data=pre_wa_graph, x="Year", y="DeathRate", label="pre policy change")
sns.lineplot(data=post_wa_graph, x="Year", y="DeathRate", label="post policy change")


plt.axvline(x=2012, color="red", linestyle="--", label="Policy Change")

plt.xlabel("Year")
plt.ylabel("Death Rate (Deaths/Total Population for each County)")
plt.title("Death Rate in WA vs Control State pre/post policy change")
plt.legend()

plt.show()

# %%
FL_WV_cleaned["DeathRate"] = FL_WV_cleaned["Deaths"] / FL_WV_cleaned["Population"]

# %%
fl = FL_WV_cleaned[FL_WV_cleaned["State"] == "FL"]
pre_fl_graph = fl[fl["Year"].isin(pre_fl)]
post_fl_graph = fl[fl["Year"].isin(post_fl)]

# plot pre_fl and post_fl in one graph
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))

sns.lineplot(data=pre_fl_graph, x="Year", y="DeathRate", label="pre policy change")
sns.lineplot(data=post_fl_graph, x="Year", y="DeathRate", label="post policy change")

wv = FL_WV_cleaned[FL_WV_cleaned["State"] == "WV"]
pre_wv_graph = wv[wv["Year"].isin(pre_fl)]
post_wv_graph = wv[wv["Year"].isin(post_fl)]

# sns.lineplot(data=pre_wv_graph, x="Year", y="DeathRate", label="control pre")
# sns.lineplot(data=post_wv_graph, x="Year", y="DeathRate", label="control post")
plt.axvline(x=2010, color="red", linestyle="--", label="Policy Change")

plt.xlabel("Year")
plt.ylabel("Death Rate (Deaths/Total Population for each County)")
plt.title("Death Rate in FL vs Control State pre/post policy change")
plt.legend()

plt.show()

# %%
TX_KS_cleaned["DeathRate"] = TX_KS_cleaned["Deaths"] / TX_KS_cleaned["Population"]

# %%
tx = TX_KS_cleaned[TX_KS_cleaned["State"] == "TX"]
pre_tx_graph = tx[tx["Year"].isin(pre)]
post_tx_graph = tx[tx["Year"].isin(post)]

# plot pre_fl and post_fl in one graph
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))

sns.lineplot(data=pre_tx_graph, x="Year", y="DeathRate", label="pre policy change")
sns.lineplot(data=post_tx_graph, x="Year", y="DeathRate", label="post policy change")

ks = TX_KS_cleaned[TX_KS_cleaned["State"] == "KS"]
pre_ks_graph = ks[ks["Year"].isin(pre)]
post_ks_graph = ks[ks["Year"].isin(post)]

# sns.lineplot(data=pre_ks_graph, x="Year", y="DeathRate", label="control pre")
# sns.lineplot(data=post_ks_graph, x="Year", y="DeathRate", label="control post")
plt.axvline(x=2007, color="red", linestyle="--", label="Policy Change")

plt.xlabel("Year")
plt.ylabel("Death Rate (Deaths/Total Population for each County)")
plt.title("Death Rate in TX  pre/post policy change")
plt.legend()

plt.show()

# %% [markdown]
# # DIFF/DIFF

# %%
import pandas as pd
import statsmodels.api as sm

wa_plus_ref = WA_OR_cleaned[WA_OR_cleaned["State"].isin(["OR", "WA"])]
wa_plus_ref = wa_plus_ref[
    wa_plus_ref["Year"].isin([2009, 2010, 2011, 2012, 2013, 2014])
]
# Create indicators for the post-policy change period and treatment state
wa_plus_ref["PostPolicy"] = (wa_plus_ref["Year"] >= 2012).astype(int)
wa_plus_ref["Treated"] = (wa_plus_ref["State"] == "WA").astype(int)

# Create interaction terms
wa_plus_ref["PostPolicy_Treated"] = wa_plus_ref["PostPolicy"] * wa_plus_ref["Treated"]
wa_plus_ref["Year_PostPolicy"] = wa_plus_ref["Year"] * wa_plus_ref["PostPolicy"]
wa_plus_ref["Year_PostPolicy_Treated"] = (
    wa_plus_ref["Year"] * wa_plus_ref["PostPolicy"] * wa_plus_ref["Treated"]
)

# Define the model
model = sm.OLS(
    wa_plus_ref["DeathRate"],
    sm.add_constant(
        wa_plus_ref[
            [
                "PostPolicy",
                "Treated",
                "PostPolicy_Treated",
                "Year",
                "Year_PostPolicy",
                "Year_PostPolicy_Treated",
            ]
        ]
    ),
)

# Fit the model
results = model.fit()

# Print regression results
print(results.summary())

# %%
# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Plot the results
sns.set(style="whitegrid")
# Separate the data for control (non-'WA') and treatment ('WA')
df_control = wa_plus_ref[wa_plus_ref["Treated"] == 0]
df_treatment = wa_plus_ref[wa_plus_ref["Treated"] == 1]
# Plot the trends
plt.figure(figsize=(10, 6))
sns.lineplot(x="Year", y="DeathRate", data=df_control, label="Control States")
sns.lineplot(x="Year", y="DeathRate", data=df_treatment, label="WA (Treatment State)")
plt.title("Diff-in-Diff Death Rate WA vs Control")
plt.axvline(x=2012, color="red", linestyle="--", label="Policy Change")
plt.xlabel("Year")
plt.ylabel("Death Rate (Counts/Total Population in a county/year)")
plt.legend()

# %% [markdown]
# # FLORIDA

# %%
fl_diff_diff = FL_WV_cleaned[FL_WV_cleaned["State"].isin(["FL", "WV"])]
fl_diff_diff = fl_diff_diff[
    fl_diff_diff["Year"].isin([2010, 2011, 2012, 2009, 2008, 2007])
]
# Create indicators for the post-policy change period and treatment state
fl_diff_diff["PostPolicy"] = (fl_diff_diff["Year"] >= 2010).astype(int)
fl_diff_diff["Treated"] = (fl_diff_diff["State"] == "FL").astype(int)

# Create interaction terms
fl_diff_diff["PostPolicy_Treated"] = (
    fl_diff_diff["PostPolicy"] * fl_diff_diff["Treated"]
)
fl_diff_diff["Year_PostPolicy"] = fl_diff_diff["Year"] * fl_diff_diff["PostPolicy"]
fl_diff_diff["Year_PostPolicy_Treated"] = (
    fl_diff_diff["Year"] * fl_diff_diff["PostPolicy"] * fl_diff_diff["Treated"]
)

# Define the model
model = sm.OLS(
    fl_diff_diff["DeathRate"],
    sm.add_constant(
        fl_diff_diff[
            [
                "PostPolicy",
                "Treated",
                "PostPolicy_Treated",
                "Year",
                "Year_PostPolicy",
                "Year_PostPolicy_Treated",
            ]
        ]
    ),
)

# Fit the model
results = model.fit()

# Print regression results
print(results.summary())

# %%
sns.set(style="whitegrid")

df_control1 = fl_diff_diff[fl_diff_diff["Treated"] == 0]
df_treatment1 = fl_diff_diff[fl_diff_diff["Treated"] == 1]
# Plot the trends
plt.figure(figsize=(10, 6))
sns.lineplot(x="Year", y="DeathRate", data=df_control1, label="Control States")
sns.lineplot(x="Year", y="DeathRate", data=df_treatment1, label="FL (Treatment State)")
plt.axvline(x=2010, color="red", linestyle="--", label="Policy Change")
plt.title("Diff-in-Diff Death Rate FL vs Control")

plt.xlabel("Year")
plt.ylabel("Death Rate (Counts/Total Population in a county/year)")
plt.legend()

# %% [markdown]
# # TEXAS

# %%
tx_diff_diff = TX_KS_cleaned[TX_KS_cleaned["State"].isin(["TX", "KS"])]
tx_diff_diff = tx_diff_diff[
    tx_diff_diff["Year"].isin([2006, 2005, 2004, 2007, 2008, 2009])
]
# Create indicators for the post-policy change period and treatment state
tx_diff_diff["PostPolicy"] = (tx_diff_diff["Year"] >= 2007).astype(int)
tx_diff_diff["Treated"] = (tx_diff_diff["State"] == "TX").astype(int)

# Create interaction terms
tx_diff_diff["PostPolicy_Treated"] = (
    tx_diff_diff["PostPolicy"] * tx_diff_diff["Treated"]
)
tx_diff_diff["Year_PostPolicy"] = tx_diff_diff["Year"] * tx_diff_diff["PostPolicy"]
tx_diff_diff["Year_PostPolicy_Treated"] = (
    tx_diff_diff["Year"] * tx_diff_diff["PostPolicy"] * tx_diff_diff["Treated"]
)

# Define the model
model = sm.OLS(
    tx_diff_diff["DeathRate"],
    sm.add_constant(
        tx_diff_diff[
            [
                "PostPolicy",
                "Treated",
                "PostPolicy_Treated",
                "Year",
                "Year_PostPolicy",
                "Year_PostPolicy_Treated",
            ]
        ]
    ),
)

# Fit the model
results = model.fit()

# Print regression results
print(results.summary())

# %%
sns.set(style="whitegrid")

df_control12 = tx_diff_diff[tx_diff_diff["Treated"] == 0]
df_treatment12 = tx_diff_diff[tx_diff_diff["Treated"] == 1]
# Plot the trends
plt.figure(figsize=(10, 6))
sns.lineplot(x="Year", y="DeathRate", data=df_control12, label="Control States")
sns.lineplot(x="Year", y="DeathRate", data=df_treatment12, label="TX (Treatment State)")
plt.axvline(x=2007, color="red", linestyle="--", label="Policy Change")
plt.title("Diff-in-Diff Death Rate TX vs Control")

plt.xlabel("Year")
plt.ylabel("Death Rate (Counts/Total Population in a county/year)")
plt.legend()
