# %% [markdown]
# ## Big Data Processing Using Dask
# 
# The purpose of this file is to replicate the opioid transaction data processing using Dask distributed processing. The python code chunks below will perform the following steps:
# 
# 1. Read a 100Gb opioid transaction dataset
# 2. Calculate the morphine equivalent for each shipment
# 3. Filter for applicable states and years
# 4. Convert dask computed results to a pandas dataframe
# 5. Save dataframe as a parquet file
# 
# *Note 1: This code requires the unzipped version of the opioid transaction data which exists in the local directory where this file is being run.*
# 
# *Note 2: The code to create a parquet file (step 5) was written but not executed for this notebook. An equivalent parquet file was created earlier using pandas chunking.*

# %%
# Import packages
import pandas as pd
from dask.distributed import Client
import dask.dataframe as dd
import pyarrow as pa
import pyarrow.parquet as pq

# %%
# Print the number of cores available
print(f"There are {os.cpu_count()} logical cores available.")

# Instantiate Dask Client 
client = Client()
client

# %%
# Reload the data
df = dd.read_csv(
    "arcos_all_washpost.tsv",
    delimiter='\t',
    dtype={
        "REPORTER_DEA_NO": "object",
        "REPORTER_BUS_ACT": "object",
        "REPORTER_NAME": "object",
        "REPORTER_ADDL_CO_INFO": "object",
        "REPORTER_ADDRESS1": "object",
        "REPORTER_ADDRESS2": "object",
        "REPORTER_CITY": "object",
        "REPORTER_STATE": "object",
        "REPORTER_ZIP": "object",
        "REPORTER_COUNTY": "object",
        "BUYER_DEA_NO": "object",
        "BUYER_BUS_ACT": "object",
        "BUYER_NAME": "object",
        "BUYER_ADDL_CO_INFO": "object",
        "BUYER_ADDRESS1": "object",
        "BUYER_ADDRESS2": "object",
        "BUYER_CITY": "object",
        "BUYER_STATE": "object",
        "BUYER_ZIP": "object",
        "BUYER_COUNTY": "object",
        "TRANSACTION_CODE": "object",
        "DRUG_CODE": "object",
        "NDC_NO": "object",
        "DRUG_NAME": "object",
        "Measure": "object",
        "MME_Conversion_Factor": "float64",
        "Dosage_Strength": "float64",
        "TRANSACTION_DATE": "object",
        "Combined_Labeler_Name": "object",
        "Reporter_family": "object",
        "CALC_BASE_WT_IN_GM": "float64",
        "DOSAGE_UNIT": "float64",
        "MME": "float64",
    },
)

# %%
# Define states to process
state_list = ['WA', 'FL', 'TX']

# Convert MME and year to correct datatypes
df["date"] = dd.to_datetime(df.TRANSACTION_DATE, format="%Y-%m-%d")
df["year"] = df.date.dt.year

df["MME_Conversion_Factor"] = dd.to_numeric(
    df["MME_Conversion_Factor"], errors="coerce"
)
# Make an estimate of total morphine equivalent shipments
df["morphine_equivalent_g"] = (df["CALC_BASE_WT_IN_GM"]) * df["MME_Conversion_Factor"]

# Drop extra variables
reduced_df = df[["year", "morphine_equivalent_g", "BUYER_STATE", "BUYER_COUNTY"]]

# Filter to applicable states
reduced_df = reduced_df[reduced_df['BUYER_STATE'].isin(state_list)]

# Filter for applicable years (2003-2015)
reduced_df = reduced_df[(reduced_df['year']>=2003)&
                        (reduced_df['year']<=2015)]

# Collapse to total shipments by each distributer.
transactions_grouped = reduced_df.groupby(
    ["year", "BUYER_STATE", "BUYER_COUNTY"]
).morphine_equivalent_g.sum()

# Print the Dask Series Structure if successful.
print(transactions_grouped)

# Generate the dask computed series
transactions_processed = transactions_grouped.compute()

# %%
# Convert dask computed results to a pandas dataframe
transactions_df = pd.DataFrame(transactions_processed)
transactions_df = transactions_df.reset_index()

# Specify the path where you want to save the Parquet file
parquet_file_path = 'transactions_WA_FL_TX.parquet'

# Convert the Pandas DataFrame to a PyArrow Table
table = pa.Table.from_pandas(transactions_df)

# Write the Table to a Parquet file
pq.write_table(table, parquet_file_path)


