import pandas as pd
import sqlite3

# Read the complete.csv dataset
df_complete = pd.read_csv('complete.csv')

# Check for missing values
missing_values = df_complete.isnull().sum()
print("Missing values in each column:\n", missing_values)

# Resolve missing values (example: fill with 0 or forward fill)
df_complete.fillna(0, inplace=True)

# Check for duplicate entries
duplicate_entries = df_complete.duplicated().sum()
print("Number of duplicate entries:", duplicate_entries)

# Remove duplicate entries
df_complete.drop_duplicates(inplace=True)

