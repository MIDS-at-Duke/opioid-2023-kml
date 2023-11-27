# Import packages
import pandas as pd
from zipfile36 import ZipFile
import pyarrow as pa
import pyarrow.parquet as pq
import io


def chunking(states_dict: list, output_file_name: str):
    """
    Objective:
        Chunks the opioid transaction data into multiple

    Args:
        states_dict (dictionary): A dictionary containing the states to be chunked

    Returns:
        A parquet file subsetted for the states provided in the argument "states" and limited to
        critical columns listed in the function below.
    """

    # File paths and parameters - This code was run in github codespaces. Change zip_file_path as needed.
    zip_file_path = "/workspaces/arcos_all_washpost.zip"
    tsv_file_name = "arcos_all_washpost.tsv"
    chunk_size = 25000
    keep_cols = [
        "BUYER_COUNTY",
        "DOSAGE_UNIT",
        "CALC_BASE_WT_IN_GM",
        "BUYER_STATE",
        "TRANSACTION_DATE",
        "DRUG_NAME",
    ]
    filtered_data = []

    # Suppressing the SettingWithCopyWarning
    pd.options.mode.chained_assignment = None  # default='warn'

    # Create a list of just the states abbreviations
    states_abv = list(states_dict.keys())

    # Open the ZIP file containing the CSV file
    with ZipFile(zip_file_path, "r") as zip_file:
        # Check if the CSV file exists in the ZIP archive
        if tsv_file_name in zip_file.namelist():
            # Open the CSV file from the ZIP archive
            with zip_file.open(tsv_file_name) as tsv_file:
                # Read the CSV file in chunks
                csv_reader = pd.read_csv(
                    tsv_file,
                    delimiter="\t",
                    chunksize=chunk_size,
                    usecols=keep_cols,
                    low_memory=False,
                )

                # Iterate through chunks of the CSV file
                for i, chunk in enumerate(csv_reader):
                    # Filter data for specific states
                    mod_chunk = chunk.loc[chunk["BUYER_STATE"].isin([states_abv])]

                    # Convert TRANSACTION_DATE to string type
                    mod_chunk["TRANSACTION_DATE"] = mod_chunk[
                        "TRANSACTION_DATE"
                    ].astype(str)

                    try:
                        # Split the date column and convert to integers
                        mod_chunk[
                            ["TransactionYear", "TransactionMonth", "TransactionDay"]
                        ] = (
                            mod_chunk["TRANSACTION_DATE"]
                            .str.split("-", expand=True)
                            .astype(int)
                        )
                    except Exception as e:
                        # Handle the error by inserting NaN values for all three date columns
                        mod_chunk[
                            ["TransactionYear", "TransactionMonth", "TransactionDay"]
                        ] = pd.NA

                    # Drop the original 'TRANSACTION_DATE' and 'TransactionDay' columns
                    mod_chunk.drop(
                        ["TRANSACTION_DATE", "TransactionDay"], axis=1, inplace=True
                    )

                    # Keep data within the specified year range
                    append_chunk = mod_chunk[
                        (mod_chunk["TransactionYear"] >= 2003)
                        & (mod_chunk["TransactionYear"] <= 2015)
                    ]

                    # Append the processed chunk to the list
                    filtered_data.append(append_chunk)

            # Concatenate all processed chunks into a single DataFrame
            selected_data = pd.concat(filtered_data, ignore_index=True)
        else:
            # Print an error message if the file is not found in the ZIP archive
            print(f"{tsv_file_name} not found in the ZIP file.")

    # Convert the Pandas DataFrame to a PyArrow Table
    table = pa.Table.from_pandas(selected_data)

    # Write the Table to a Parquet file
    pq.write_table(table, output_file_name)


def read_parquet(zip_name: str, file_name: str):
    """
    Objective:
        Read a parquet file from within a zipped folder

    Args:
        zip_name (string): zip folder name. zip folder must be located within the same
        directory as this file. Must end in ".zip"

        file_name (string): parquet file name located within the zipped folder. Must
        end in ".parquet"

    Returns:
        pandas_df (pandas dataframe): Pandas dataframe containing the contents
        of the parquet file.
    """

    # File paths and names
    zip_file_path = zip_name
    pq_file_name = file_name

    # Open the ZIP file containing the Parquet file
    with ZipFile(zip_file_path, "r") as zip_file:
        # Check if the Parquet file exists in the ZIP archive
        if pq_file_name in zip_file.namelist():
            # Open the Parquet file from the ZIP archive
            with zip_file.open(pq_file_name) as pq_file:
                # Read the Parquet file into a BytesIO buffer
                # This is necessary because PyArrow requires a file-like object
                pq_buffer = io.BytesIO(pq_file.read())

                # Read the Parquet table from the buffer
                # PyArrow reads the data into a Table format
                table = pq.read_table(pq_buffer)

                # Convert the PyArrow Table into a Pandas DataFrame
                # This allows for easier manipulation and analysis of the data
                pandas_df = table.to_pandas()
        else:
            # Print an error message if the Parquet file is not found in the ZIP archive
            print(f"{pq_file_name} not found in the ZIP file.")

    return pandas_df


def county_grouping(states_dict: dict, transactions_df: pd.DataFrame):
    """
    Objective:
        Sum opioid transaction data for each state included in transactions_df.

    Args:
        states_dict (dictionary): A dictionary containing the state abbreviation (key) and state name (value)

        transactions_df (pandas DataFrame): Dataframe output from read_parquet() function. Contains all
        opioid transaction data for the states included in states_dict

    Returns:
        A parquet file for each state in states_dict, with the opioid transaction data grouped by county.
        Naming convention is `{state_name}.parquet` and saved in the current working directory.
    """

    for state in states_dict.keys():
        transactions_subset = transactions_df[
            transactions_df["BUYER_STATE"] == f"{state}"
        ]
        # Group the DataFrame by specified columns and sum specified columns
        transactions_grouped = transactions_subset.groupby(
            [
                "BUYER_STATE",
                "BUYER_COUNTY",
                "DRUG_NAME",
                "TransactionYear",
                "TransactionMonth",
            ]
        ).sum(["CALC_BASE_WT_IN_GM", "DOSAGE_UNIT"])
        transactions_grouped = transactions_grouped.reset_index()

        # Save Florida grouped data to parquet file
        file_name = states_dict[state] + ".parquet"
        transactions_grouped.to_parquet(file_name)


if __name__ == "__main__":
    # Specify The state abbreviation and state name in the dictionary
    states_dict = {"FL": "florida", "TX": "texas", "WA": "washington"}

    # Specify the chunking output file name
    output_file_name = "opioid_transactions.parquet"

    # Chunk opioid tranactions data by state abbreviation
    chunking(states_dict, output_file_name)

    # Read parquet file from zipped folder
    zip_name = "opioid_transactions.zip"
    parquet_name = "opioid_transactions.parquet"
    transactions_df = read_parquet(zip_name, parquet_name)

    # Sum opioid transactions by county in each state
    county_grouping(states_dict, transactions_df)
