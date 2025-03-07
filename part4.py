import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns

df_complete = pd.read_csv('complete.csv')
print(df_complete.head())

# Establish a connection to the SQLite database
conn = sqlite3.connect("covid_database.db")
cursor = conn.cursor()

# Check for missing values
missing_values = df_complete.isnull().sum()
print("Missing values in each column:\n", missing_values)

# disregard the records containing missing values
###df_complete = df_complete.dropna() # deletes all records that contain NaN
###df_complete = df_complete.dropna(subset=['Confirmed', 'Deaths', 'Recovered']) # deletes records when these columns contain NaN
###print(df_complete.head())

# fill in 0 for missing values
###df_complete.fillna(
###    {col: 0 if df_complete[col].dtype in ['int64', 'float64'] else '' for col in df_complete.columns}, 
###    inplace=True
###) # fills in 0 for numeric columns and en empty string else
###df_complete.fillna(0, inplace=True) # fills in 0 for all columns
###print(df_complete.head())

# fill in mean for missing values of numeric columns and 'Unkown' for string columns
###df_complete.fillna({
###    'Province.State': 'Unknown',
###    'Confirmed': df_complete['Confirmed'].mean(),
###    'Deaths': df_complete['Deaths'].mean(),
###    'Recovered': df_complete['Recovered'].mean(),  
###    'Active': df_complete['Active'].mean(), 
###}, inplace=True)
#print(df_complete.head())

# Fill in 'Unknown' for missing values in 'Province.State'
df_complete['Province.State'].fillna('Unknown', inplace=True)

# Forward fill for other columns
df_complete.ffill(inplace=True)

# Backward fill for other columns
df_complete.bfill(inplace=True)

print(df_complete.head())

# fill in the next available value
###df_complete['Province.State'].fillna('Unknown', inplace=True)
###df_complete.fillna(method='bfill', inplace=True)
#print(df_complete.head())

# check for duplicate entries
duplicates = df_complete[df_complete.duplicated()] # all columns
###duplicates = df_complete[df_complete.duplicated(subset=['Date', 'Country.Region', 'Confirmed'])] # subset of columns
print(duplicates) # only prints the >2 occurrence
print(f"Total duplicate rows: {df_complete.duplicated().sum()}")

# handle duplicate entries
df_complete.drop_duplicates(inplace=True) # keep first occurrence
###df_complete = df_complete.drop_duplicates(subset=['Date', 'Country', 'Confirmed']) # delete duplicates based on a subset

df_complete.to_sql("complete", conn, if_exists="replace", index=False)

# ---- Fetch Country Wise Data Series ----
def fetch_country_data(csv_path, country):
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df_country = df[df["Country.Region"] == country].copy()
    df_country["Active_per_capita"] = df_country["Active"] / df_country["Confirmed"].max()
    df_country["Deaths_per_capita"] = df_country["Deaths"] / df_country["Confirmed"].max()
    df_country["Recovered_per_capita"] = df_country["Recovered"] / df_country["Confirmed"].max()

    return df_country

csv_path = "complete.csv"  # Replace with your actual CSV file path
selected_country = "United Kingdom"
df_countryss = fetch_country_data(csv_path, selected_country)

# ---- Fetch Global Time Series Data ----
def fetch_global_data(conn):
    query_global = """
    SELECT Date, Confirmed, Deaths, Recovered, Active
    FROM day_wise
    ORDER BY Date;
    """
    df_global = pd.read_sql_query(query_global, conn)
    df_global["Date"] = pd.to_datetime(df_global["Date"])
    return df_global

# ---- Function to Plot COVID-19 Trends ----
def covid_trends(df, country):
    fig, axes = plt.subplots(3, 1, figsize=(15, 9))

    axes[0].plot(df["Date"], df["Confirmed"], color="blue", label="Confirmed Cases")
    axes[0].set_title(f"{country}: Daily Confirmed COVID-19 Cases")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Cases")
    axes[0].legend()

    axes[1].plot(df["Date"], df["Deaths"], color="red", label="Deaths")
    axes[1].set_title(f"{country}: Daily COVID-19 Deaths")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Deaths")
    axes[1].legend()

    axes[2].plot(df["Date"], df["Recovered"], color="green", label="Recoveries")
    axes[2].set_title(f"{country}: Daily Recoveries from COVID-19")
    axes[2].set_xlabel("Date")
    axes[2].set_ylabel("Recoveries")
    axes[2].legend()

    plt.tight_layout()
    plt.show()

# Example usage
covid_trends(df_countryss, selected_country)

country = "France"
query_country = f"""
    SELECT d.Date, c."Country.Region", d.Confirmed, d.Deaths, d.Recovered, d.Active, w.Population
    FROM day_wise d
    JOIN worldometer_data w ON w."Country.Region" = '{country}'
    JOIN country_wise c ON c."Country.Region" = w."Country.Region"
    WHERE c."Country.Region" = '{country}'
    ORDER BY d.Date;
    """
df_country = pd.read_sql_query(query_country, conn)
df_country["Date"] = pd.to_datetime(df_country["Date"])

# Aggregate data per country over time
df_country_grouped = df_country.groupby(["Country.Region", "Date"]).sum().reset_index()

# ---- Fetch and Group by US County ----
query_county = """
SELECT Province_State, Date, Confirmed, Deaths
FROM usa_county_wise;
"""
df_county = pd.read_sql_query(query_county, conn)
df_county["Date"] = pd.to_datetime(df_county["Date"])

# Aggregate data per US county over time
df_county_grouped = df_county.groupby(["Province_State", "Date"]).sum().reset_index()

# ---- Plot Country-Level Data ----
plt.figure(figsize=(12, 6))
sns.lineplot(data=df_country_grouped, x="Date", y="Confirmed", hue="Country.Region", legend=False)
plt.title("COVID-19 Confirmed Cases per Country Over Time")
plt.xlabel("Date")
plt.ylabel("Confirmed Cases")
plt.xticks(rotation=45)
plt.show()

# ---- Plot US County-Level Data ----
plt.figure(figsize=(12, 6))
sns.lineplot(data=df_county_grouped, x="Date", y="Confirmed", hue="Province_State", legend=False)
plt.title("COVID-19 Confirmed Cases per US State Over Time")
plt.xlabel("Date")
plt.ylabel("Confirmed Cases")
plt.xticks(rotation=45)
plt.show()

# Close the database connection
conn.close()