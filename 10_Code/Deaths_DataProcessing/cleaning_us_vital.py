import pandas as pd
import os

# Change file path here
import pandas as pd
import os

directory_path = "C:\\Users\\khsqu\\OneDrive\\Documents\\PDS\\opioid-2023-kml\\cleaning_us_vital\\US_VitalStatistics\\"
merged_data = pd.DataFrame()

for filename in os.listdir(directory_path):
    if filename.endswith(".txt"):
        year = filename.split(",")[1].split(".")[0].strip()  # Extract the year
        print(year)
        file_path = os.path.join(directory_path, filename)

        df = pd.read_csv(file_path, delimiter="\t")
        print(df)
        # Add a 'Year' column to the DataFrame
        df["Year"] = year
        print(df)
        # Append the data to the merged_data DataFrame
        print(filename)
        merged_data = merged_data._append(df, ignore_index=True)


# Save the merged DataFrame to a CSV file
merged_data.to_csv(
    "C:\\Users\\khsqu\\OneDrive\\Documents\\PDS\\opioid-2023-kml\\cleaning_us_vital\\merged_data.csv",
    index=False,
)
