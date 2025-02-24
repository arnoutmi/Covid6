import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns

# Connect to SQLite Database
conn = sqlite3.connect("covid_database.db")

# ---- Fetch Country-Specific Data (United States) ----
query = """
SELECT c."Country.Region", c.Active, c.Deaths, c.Recovered, w.population
FROM country_wise c
JOIN worldometer_data w ON c."Country.Region" = w."Country.Region"
WHERE c."Country.Region" = 'United States';
"""

df_country = pd.read_sql_query(query, conn)

# ---- Fetch Global Time Series Data ----
query_global = """
SELECT Date, Confirmed, Deaths, Recovered, Active
FROM day_wise
ORDER BY Date;
"""

df_global = pd.read_sql_query(query_global, conn)

# Convert Date column to datetime format
df_global["Date"] = pd.to_datetime(df_global["Date"])

# Close connection
conn.close()

# ---- Function to Plot COVID-19 Trends ----
def covid_trends(df):
    fig, axes = plt.subplots(3, 1, figsize=(15, 9))

    axes[0].plot(df["Date"], df["Confirmed"], color="blue", label="Confirmed Cases")
    axes[0].set_title("Global Daily Confirmed COVID-19 Cases")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Cases")
    axes[0].legend()

    axes[1].plot(df["Date"], df["Deaths"], color="red", label="Deaths")
    axes[1].set_title("Global Daily COVID-19 Deaths")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Deaths")
    axes[1].legend()

    axes[2].plot(df["Date"], df["Recovered"], color="green", label="Recoveries")
    axes[2].set_title("Global Daily Recoveries from COVID-19")
    axes[2].set_xlabel("Date")
    axes[2].set_ylabel("Recoveries")
    axes[2].legend()

    plt.tight_layout()
    plt.show()

# ---- Function to Estimate SIR Model Parameters ----
def estimation_of_parameters(df):
    N = 17000000  # Example population size
    beta_estimates = []
    gamma_estimates = []
    mu_estimates = []
    R0_estimates = []

    for i in range(1, len(df)):
        new_cases = df["Confirmed"].iloc[i] - df["Confirmed"].iloc[i - 1]
        new_deaths = df["Deaths"].iloc[i] - df["Deaths"].iloc[i - 1]
        new_recovered = df["Recovered"].iloc[i] - df["Recovered"].iloc[i - 1]

        I_t = df["Active"].iloc[i]
        R_t = df["Recovered"].iloc[i]
        D_t = df["Deaths"].iloc[i]
        S_t = N - I_t - R_t - D_t  

        if I_t > 0:
            beta = (new_cases * N) / (S_t * I_t)
            beta_estimates.append(beta)

            gamma = new_recovered / I_t if I_t > 0 else 0
            gamma_estimates.append(gamma)

            mu = new_deaths / I_t if I_t > 0 else 0
            mu_estimates.append(mu)

            if gamma > 0:
                R0 = beta / gamma
                R0_estimates.append(R0)

    avg_beta = np.mean(beta_estimates) if beta_estimates else 0
    avg_gamma = np.mean(gamma_estimates) if gamma_estimates else 0
    avg_mu = np.mean(mu_estimates) if mu_estimates else 0
    avg_R0 = np.mean(R0_estimates) if R0_estimates else 0

    print(f"Estimated Beta: {avg_beta}")
    print(f"Estimated Gamma: {avg_gamma}")
    print(f"Estimated Mu: {avg_mu}")
    print(f"Estimated R0 (Basic Reproduction Number): {avg_R0}")

    plt.figure(figsize=(10, 6))
    plt.plot(df["Date"].iloc[1:], R0_estimates, label="Estimated R0", color="purple")
    plt.title("Estimated Basic Reproduction Number (R₀) Over Time")
    plt.xlabel("Date")
    plt.ylabel("R₀")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ---- Function to Plot Active vs Recovered vs Deaths ----
def active_vs_recovered_vs_deaths_plot(df):
    plt.figure(figsize=(10, 6))
    plt.stackplot(df['Date'], df['Active'], df['Recovered'], df['Deaths'], labels=['Active', 'Recovered', 'Deaths'], alpha=0.75)
    plt.title("Active vs. Recovered vs. Deaths Over Time")
    plt.xlabel("Date")
    plt.ylabel("Count")
    plt.legend(loc="lower left")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# ---- Function to Calculate Growth Rate of Cases ----
def growth_rate_cases(df):
    df['Growth rate'] = df['Confirmed'].pct_change() * 100
    plt.figure(figsize=(10, 6))
    plt.plot(df['Date'], df['Growth rate'], color="purple", label="Daily Growth Rate of New Cases")
    plt.title("Daily Growth Rate of New COVID-19 Cases")
    plt.xlabel("Date")
    plt.ylabel("Growth Rate (%)")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# ---- Run Functions ----
covid_trends(df_global)  # Global Trends
estimation_of_parameters(df_global)  # SIR Model Estimations
active_vs_recovered_vs_deaths_plot(df_global)  # Stacked Plot
growth_rate_cases(df_global)  # Growth Rate Analysis
