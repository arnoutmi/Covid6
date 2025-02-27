import pandas as pd
import sqlite3

df_complete = pd.read_csv('complete.csv')
print(df_complete.head())

missing_values = df_complete.isnull().sum()
print("Missing values in each column:\n", missing_values)

# disregard the records containing missing values
###df_complete = df_complete.dropna() # deletes all records that contain NaN
###df_complete = df_complete.dropna(subset=['Confirmed', 'Deaths', 'Recovered']) # deletes records when these columns contain NaN

# fill in 0 for missing values
df_complete.fillna(
    {col: 0 if df_complete[col].dtype in ['int64', 'float64'] else '' for col in df_complete.columns}, 
    inplace=True
) # fills in 0 for numeric columns and en empty string else
###df_complete.fillna(0, inplace=True) # fills in 0 for all columns
print(df_complete.head())

# fill in mean for missing values of numeric columns and 'Unkown' for string columns
###df_complete.fillna({
###    'Province.State': 'Unknown',
###    'Confirmed': df_complete['Confirmed'].mean(),
###    'Deaths': df_complete['Deaths'].mean(),
###    'Recovered': df_complete['Recovered'].mean(),  
###    'Active': df_complete['Active'].mean(), 
###}, inplace=True)
#print(df_complete.head())


# fill in the previous value
###df_complete.fillna(method='ffill', inplace=True)
#print(df_complete.head())

# fill in the next available value
###df_complete.fillna(method='bfill', inplace=True)
#print(df_complete.head())

# check for duplicate entries
duplicates = df_complete[df_complete.duplicated()] # all columns
###duplicates = df_complete[df_complete.duplicated(subset=['Date', 'Country.Region', 'Confirmed'])] # subset of columns
print(duplicates) # only prints the >2 occurence
print(f"Total duplicate rows: {df_complete.duplicated().sum()}")

# handle duplicate entries
df_complete.drop_duplicates(inplace=True) # keep first occurence
###df_complete = df_complete.drop_duplicates(subset=['Date', 'Country', 'Confirmed']) # delete duplicates based on a subset
