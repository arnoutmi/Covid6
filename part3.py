import sqlite3

# Connect to the database file
conn = sqlite3.connect("covid_database.db")  # Change this to the actual filename
cursor = conn.cursor()

# Check available tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print("Tables in database:", tables)

# Get column names for a table
table_name = "country_wise"  # Change to each table name
cursor.execute(f"PRAGMA table_info({table_name});")
columns = cursor.fetchall()
print(f"Columns in {table_name}:", columns)

import pandas as pd

# Example: Get data for a specific country (change country name)
country = "United States"  # Change this to any country you want
query = f"""
SELECT d.date, c.country, c.active, c.deaths, c.recovered, w.population
FROM day_wise d
JOIN country_wise c ON d.date = c.date
JOIN worldometer_data w ON c.country = w.country
WHERE c.country = '{country}'
ORDER BY d.date;
"""

df = pd.read_sql_query(query, conn)
print(df.head())

import matplotlib.pyplot as plt

# Convert date column to datetime format
df['date'] = pd.to_datetime(df['date'])

# Compute the percentage
df['active_perc'] = df['active'] / df['population'] * 100
df['deaths_perc'] = df['deaths'] / df['population'] * 100
df['recovered_perc'] = df['recovered'] / df['population'] * 100

# Plot
plt.figure(figsize=(12, 6))
plt.plot(df['date'], df['active_perc'], label="Active Cases (%)", color='blue')
plt.plot(df['date'], df['deaths_perc'], label="Deaths (%)", color='red')
plt.plot(df['date'], df['recovered_perc'], label="Recovered (%)", color='green')

plt.xlabel("Date")
plt.ylabel("Percentage of Population")
plt.title(f"Covid-19 Cases as Percentage of Population in {country}")
plt.legend()
plt.grid()
plt.show()
