# %%
import pandas as pd
import numpy as np

# %%
# read in the csv
overdoses = pd.read_csv("cleaning_us_vital\merged_data.csv")
overdoses.head(10)

# %%
overdoses.shape

# %%
overdoses["County"].head(10)

# %%
# Here we subset by the policy states. We will need to add the control states.
selected_states = ["FL", "TX", "WA", "WV", "TN", "OH", "NE", "WY", "KS", "OR", "IL"]

filtered_df = overdoses[overdoses["County"].notna()]
filtered_df = filtered_df[filtered_df["County"].str.endswith(tuple(selected_states))]

filtered_df.head(10)

# %%
# Checking that it was reduced
filtered_df.shape

# %%
filtered_df["County"].unique().shape

# %%
# Put all the different categories into 1 series for further filtering
causes_of_death = filtered_df["Drug/Alcohol Induced Cause"].unique()
causes_of_death

# %%
filtered_df = filtered_df.drop(columns=["Notes", "Drug/Alcohol Induced Cause Code"])

# %%
filtered_df.shape

# %%
## We see a county which contains overdoses in some years and others which don't.
print(filtered_df[filtered_df["County"] == "Pike County, OH"])

# %%
filtered_df["Drug/Alcohol Induced Cause"] = filtered_df[
    "Drug/Alcohol Induced Cause"
].replace(
    to_replace=[
        "Drug poisonings (overdose) Unintentional (X40-X44)",
        "Drug poisonings (overdose) Suicide (X60-X64)",
        "Drug poisonings (overdose) Undetermined (Y10-Y14)",
        "All other drug-induced causes",
    ],
    value="Drug Causes",
)

filtered_df[filtered_df["County"] == "Broward County, FL"]

# %%
filtered_df["Deaths"] = pd.to_numeric(filtered_df["Deaths"], errors="coerce")

# Filter rows where the cause is 'Drug Causes'
drug_causes_df = filtered_df[filtered_df["Drug/Alcohol Induced Cause"] == "Drug Causes"]

# Group by 'County' and 'Year', then sum the 'Deaths'
result_df = (
    drug_causes_df.groupby(["County", "Year"]).agg({"Deaths": "sum"}).reset_index()
)
result_df[result_df["County"] == "Broward County, FL"]

# %%
# 595 total counties
filtered_df["County"].unique().size

# %% [markdown]
# We have 1934 rows entries of observations. Given we have 595 counties, we should have data for each year for each county. Our range is 2003-2015 which is 12 years. 595 * 12 years = 7,140 total observations for drug deaths.

# %%
import os

all_counties_years = filtered_df[["County", "Year"]].drop_duplicates()

# Merge the original DataFrame with the complete set
merged_df = pd.merge(all_counties_years, result_df, on=["County", "Year"], how="left")

# Identify rows where there is no drug death entry
no_drug_death_entries = merged_df[merged_df["Deaths"].isna()]

# Print the result
print("Years with no drug death entry for each county:")
print(no_drug_death_entries[["County", "Year"]])

output_directory = "state_text_files"

# Create the directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

output_file_path = os.path.join("state_text_files", "counties_with_missingDeaths.txt")
no_drug_death_entries.to_csv(output_file_path, sep="\t", index=False)

# %%
output_file_path1 = os.path.join("state_text_files", "drug_deaths.txt")
result_df.to_csv(output_file_path1, sep="\t", index=False)
result_df.head(10)

# %%
unique_counties = merged_df["County"].unique()
unique_counties_df = pd.DataFrame({"County": unique_counties})

output_file_path2 = os.path.join("state_text_files", "all_counties.txt")
unique_counties_df.to_csv(output_file_path2, sep="\t", index=False)

# %% [markdown]
# ### To-Do & Problems

# %% [markdown]
# > Still need to fill in missing data or fix missing data based on what we decide
#
# > Convert this into  .py file once complete
