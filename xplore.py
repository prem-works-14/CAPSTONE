import pandas as pd

# Path to your CSV file
csv_path = "C:/Users/Dell/Downloads/climax_training_setup/data/era5.csv"

# Show only first few rows and selected columns
df_preview = pd.read_csv(csv_path, nrows=1000)  # reads only 5 rows
print("🔍 Columns available in the CSV:\n", df_preview.columns)

# If you want to show specific columns, e.g., 'valid_time', 't', 'z':
print("\n🧪 Sample Data from Columns:")
print(df_preview[["valid_time", "t", "z"]])  # only these columns shown
