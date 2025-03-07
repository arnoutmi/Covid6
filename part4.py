import pandas as pd
import sqlite3
import matplotlib.pyplot as plt

df_complete = pd.read_csv('complete.csv')
print(df_complete.head())

conn = sqlite3.connect("covid_database.db")
cursor = conn.cursor()

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

df_complete.to_sql("complete", conn, if_exists="replace", index=False)

# ---- Fetch Country Wise Data Series ----
def fetch_country_data(conn, country='France'):
    query = f"""
    SELECT d.Date, c."Country.Region", d.Confirmed, d.Deaths, d.Recovered, d.Active, w.Population
    FROM day_wise d
    JOIN worldometer_data w ON w."Country.Region" = c."Country.Region"
    JOIN country_wise c ON c."Country.Region" = '{country}'
    WHERE c."Country.Region" = '{country}'
    ORDER BY d.Date;
    """
    df_country = pd.read_sql_query(query, conn)
    df_country["Date"] = pd.to_datetime(df_country["Date"])

    df_country["Active_per_capita"] = df_country["Active"] / df_country["Population"]
    df_country["Deaths_per_capita"] = df_country["Deaths"] / df_country["Population"]
    df_country["Recovered_per_capita"] = df_country["Recovered"] / df_country["Population"]

    return df_country


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
def covid_trends(df):
    fig, axes = plt.subplots(3, 1, figsize=(15, 9))

    axes[0].plot(df["Date"], df["Active_per_capita"], color="blue", label="Active Cases per Capita")
    axes[0].set_title("Daily Active COVID-19 Cases (Per Capita)")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Cases per Capita")
    axes[0].legend()

    axes[1].plot(df["Date"], df["Deaths_per_capita"], color="red", label="Deaths per Capita")
    axes[1].set_title("Daily COVID-19 Deaths (Per Capita)")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Deaths per Capita")
    axes[1].legend()

    axes[2].plot(df["Date"], df["Recovered_per_capita"], color="green", label="Recoveries per Capita")
    axes[2].set_title("Daily COVID-19 Recoveries (Per Capita)")
    axes[2].set_xlabel("Date")
    axes[2].set_ylabel("Recoveries per Capita")
    axes[2].legend()

    plt.tight_layout()
    plt.show()

df_country = fetch_country_data(conn, country="France")
covid_trends(df_country)

# worldwide data per country using group by
df_global_grouped = df_complete.groupby(["Date", "Country.Region"]).agg({
    "Confirmed": "sum",
    "Deaths": "sum",
    "Recovered": "sum",
    "Active": "sum"
}).reset_index()

# USA county data per state using group by
county_query = """
    SELECT "Province.State", "Country.Region", Date, Confirmed, Deaths, Recovered, Active
    FROM complete
    WHERE "Country.Region" = 'US';
"""
df_county_data = pd.read_sql_query(county_query, conn)

df_county_grouped = df_county_data.groupby(["Date", "Country.Region", "Province.State"]).agg({
    "Confirmed": "sum",
    "Deaths": "sum",
    "Recovered": "sum",
    "Active": "sum"
}).reset_index()

conn.close()