# This code checks how many 0 and 1 there are in the HAS_GOING_CONCERN_MODIFICATION column
import pandas as pd
from pathlib import Path

# Define file path
data_folder = Path("data")
#file_path = data_folder / "features_non_zero_2639.xlsx"
file_path = data_folder / "features.xlsx"

# Read Excel file
df = pd.read_excel(file_path)

# Look for specific column
target_column = "HAS_GOING_CONCERN_MODIFICATION"

# Verify column exists
if target_column in df.columns:
    # Count 0s and 1s in the target column
    count_0 = (df[target_column] == 0).sum()
    count_1 = (df[target_column] == 1).sum()
    
    print(f"Count of 0s in column ('{target_column}'): {count_0}")
    print(f"Count of 1s in column ('{target_column}'): {count_1}")
    print(f'Percentage of positives is {(count_1/(count_0 + count_1)) * 100:.2f}%')
else:
    print(f"Error: Column '{target_column}' not found in the dataset.")
    print("Available columns:")
    print(df.columns.tolist())