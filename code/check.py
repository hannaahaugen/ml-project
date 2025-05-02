# this code just checks how many 0 and 1 there is in the data set
import pandas as pd
from pathlib import Path

# Define file path
data_folder = Path("data")
#file_path = data_folder / "features_non_zero_2639.xlsx"
file_path = data_folder / "features.xlsx"

# Read Excel file
df = pd.read_excel(file_path)

# Get last column name
last_column = df.columns[-1]

# Count 0s and 1s in the last column
count_0 = (df[last_column] == 0).sum()
count_1 = (df[last_column] == 1).sum()

print(f"Count of 0s in last column ('{last_column}'): {count_0}")
print(f"Count of 1s in last column ('{last_column}'): {count_1}")

print(f'Percentage of positives is {(count_1/(count_0 + count_1)) * 100}%')