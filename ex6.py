import pandas as pd
import dask.dataframe as dd
import pyarrow as pa
import pyarrow.parquet as pq
import os

# Print the number of logical cores available
print(f"There are {os.cpu_count()} logical cores available.")

# Instantiate Dask Client 
client = Client()
client

# Load data using Dask
df = dd.read_csv(
    "arcos_all_washpost.tsv",
    delimiter='\t',
    dtype={
        "REPORTER_DEA_NO": "object",
        "REPORTER_BUS_ACT": "object",
        "REPORTER_NAME": "object",
        # ... (other columns) ...
        "MME": "float64",
    },
)

# Convert MME and year to correct datatypes
df["date"] = dd.to_datetime(df.TRANSACTION_DATE, format="%Y-%m-%d")
df["year"] = df.date.dt.year
df["MME_Conversion_Factor"] = dd.to_numeric(df["MME_Conversion_Factor"], errors="coerce")

# Estimate morphine equivalent for each shipment
df["morphine_equivalent_g"] = df["CALC_BASE_WT_IN_GM"] * df["MME_Conversion_Factor"]

# Select relevant columns
reduced_df = df[["year", "morphine_equivalent_g", "BUYER_STATE", "BUYER_COUNTY"]]

# Define states to process
state_list = ['WA', 'FL', 'TX']

# Filter to applicable states
reduced_df = reduced_df[reduced_df['BUYER_STATE'].isin(state_list)]

# Filter for applicable years (2003-2015)
reduced_df = reduced_df[(reduced_df['year'] >= 2003) & (reduced_df['year'] <= 2015)]

# Group by year, state, and county, and sum morphine equivalents
transactions_grouped = reduced_df.groupby(
    ["year", "BUYER_STATE", "BUYER_COUNTY"]
).morphine_equivalent_g.sum()

# Print the Dask Series Structure if successful.
print(transactions_grouped)

# Compute the Dask Series
transactions_processed = transactions_grouped.compute()

# Convert Dask results to a Pandas DataFrame
transactions_df = pd.DataFrame(transactions_processed).reset_index()

# Specify the path where you want to save the Parquet file
parquet_file_path = 'transactions_WA_FL_TX.parquet'

# Convert the Pandas DataFrame to a PyArrow Table
table = pa.Table.from_pandas(transactions_df)

# Write the Table to a Parquet file
pq.write_table(table, parquet_file_path)
