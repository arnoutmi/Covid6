import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime
import plotly.express as px
import numpy as np
# -------- PAGE CONFIG -------- #
st.set_page_config(page_title="COVID-19 Data Dashboard", layout="wide")

# -------- DATABASE FETCHING AND PROCESSING -------- #

def filter_data_by_date(df, start_date, end_date):
    return df[(df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))]

def fetch_country_list(db_path="covid_database.db"):
    conn = sqlite3.connect(db_path)
    query = "SELECT DISTINCT `Country.Region` FROM covid_data ORDER BY `Country.Region`"
    countries = pd.read_sql(query, conn)['Country.Region'].tolist()
    conn.close()
    return countries

def fetch_region_list(db_path="covid_database.db"):
    conn = sqlite3.connect(db_path)
    query = "SELECT DISTINCT Region FROM covid_data ORDER BY Region"
    regions = pd.read_sql(query, conn)['Region'].dropna().tolist()
    conn.close()
    return regions

def fetch_country_data(country, db_path="covid_database.db"):
    conn = sqlite3.connect(db_path)
    query = "SELECT Date, Confirmed, Active, Recovered, Deaths FROM covid_data WHERE `Country.Region` = ?"
    df_country = pd.read_sql(query, conn, params=(country,))
    conn.close()
    df_country["Date"] = pd.to_datetime(df_country["Date"])
    return df_country.sort_values("Date")

def fetch_global_data(db_path="covid_database.db"):
    conn = sqlite3.connect(db_path)
    query = """
        SELECT Date, SUM(Confirmed) AS Confirmed, SUM(Active) AS Active, SUM(Recovered) AS Recovered, SUM(Deaths) AS Deaths
        FROM covid_data
        GROUP BY Date
        ORDER BY Date;
    """
    df_global = pd.read_sql(query, conn)
    conn.close()
    df_global["Date"] = pd.to_datetime(df_global["Date"])
    return df_global

def fetch_region_data(region, db_path="covid_database.db"):
    conn = sqlite3.connect(db_path)
    query = """
        SELECT Date, SUM(Confirmed) AS Confirmed, SUM(Active) AS Active, SUM(Recovered) AS Recovered, SUM(Deaths) AS Deaths
        FROM covid_data
        WHERE Region = ?
        GROUP BY Date
        ORDER BY Date;
    """
    df_region = pd.read_sql(query, conn, params=(region,))
    conn.close()
    df_region["Date"] = pd.to_datetime(df_region["Date"])
    return df_region

def estimate_parameters_for_country(df, country):
    """
    Estimates β, μ, R₀ for a country, using real data.
    """
    df = df.sort_values("Date").reset_index(drop=True)

    required_columns = {"Date", "Active", "Recovered", "Deaths", "Confirmed", "Population"}
    if not required_columns.issubset(df.columns):
        print(f"Missing columns for {country}: {required_columns - set(df.columns)}")
        return None

    N = df["Population"].iloc[0] if "Population" in df.columns else 1e6  # Default pop
    gamma = 1 / 4.5  # Fixed recovery rate

    beta_estimates, mu_estimates, R0_estimates = [], [], []

    for i in range(1, len(df)):
        new_cases = df["Confirmed"].iloc[i] - df["Confirmed"].iloc[i - 1]
        new_deaths = df["Deaths"].iloc[i] - df["Deaths"].iloc[i - 1]
        new_recovered = df["Recovered"].iloc[i] - df["Recovered"].iloc[i - 1]

        I_t = df["Active"].iloc[i]
        R_t = df["Recovered"].iloc[i]
        D_t = df["Deaths"].iloc[i]
        S_t = N - I_t - R_t - D_t

        if I_t > 0 and S_t > 0:
            mu = new_deaths / I_t if I_t > 0 else 0
            mu_estimates.append(mu)

            beta = ((new_cases + mu * I_t + gamma * I_t) * N) / (S_t * I_t)
            beta_estimates.append(beta)

            R0 = beta / gamma
            R0_estimates.append(R0)

    return {
        "Country": country,
        "Beta": np.mean(beta_estimates) if beta_estimates else 0,
        "Mu": np.mean(mu_estimates) if mu_estimates else 0,
        "R0": np.mean(R0_estimates) if R0_estimates else 0
    }
    """
    Estimates key epidemiological parameters for a country:
    Returns a dictionary with country name and estimated parameters.
    """
    df = df.sort_values("Date").reset_index(drop=True)
    
    required_columns = {"Date", "Active", "Recovered", "Deaths", "Confirmed", "Population"}
    if not required_columns.issubset(df.columns):
        print(f"Dataframe is missing columns for {country}: {required_columns - set(df.columns)}")
        return None
    
    N = df["Population"].iloc[0] if "Population" in df.columns else 1e6

    gamma = 1 / 4.5  # Fixed recovery rate

    beta_estimates, mu_estimates, R0_estimates, alpha_estimates = [], [], [], []

    for i in range(1, len(df)):
        new_cases = df["Confirmed"].iloc[i] - df["Confirmed"].iloc[i - 1]
        new_deaths = df["Deaths"].iloc[i] - df["Deaths"].iloc[i - 1]
        new_recovered = df["Recovered"].iloc[i] - df["Recovered"].iloc[i - 1]

        I_t = df["Active"].iloc[i]
        R_t = df["Recovered"].iloc[i]
        D_t = df["Deaths"].iloc[i]
        S_t = N - I_t - R_t - D_t  

        if I_t > 0 and S_t > 0:
            mu = new_deaths / I_t if I_t > 0 else 0
            mu_estimates.append(mu)

            beta = ((new_cases + mu * I_t + gamma * I_t) * N) / (S_t * I_t)
            beta_estimates.append(beta)

            alpha = (gamma * I_t - new_recovered) / R_t if R_t > 0 else 0
            alpha_estimates.append(alpha)

            R0 = beta / gamma
            R0_estimates.append(R0)

    # Return averaged parameters
    return {
        "Country": country,
        "Beta": np.mean(beta_estimates) if beta_estimates else 0,
        "Alpha": np.mean(alpha_estimates) if alpha_estimates else 0,
        "Mu": np.mean(mu_estimates) if mu_estimates else 0,
        "R0": np.mean(R0_estimates) if R0_estimates else 0
    }

def estimate_parameters_all_countries(db_path="covid_database.db"):
    conn = sqlite3.connect(db_path)
    query = """
    SELECT cd.*, wd.Population
    FROM covid_data cd
    LEFT JOIN worldometer_data wd
    ON cd.`Country.Region` = wd.`Country.Region`
    """
    df = pd.read_sql(query, conn)
    conn.close()

    countries = df['Country.Region'].unique()
    parameter_list = []

    for country in countries:
        df_country = df[df['Country.Region'] == country].copy()
        if len(df_country) < 10:  # Skip small datasets
            continue
        params = estimate_parameters_for_country(df_country, country)
        if params:
            parameter_list.append(params)

    return pd.DataFrame(parameter_list)
    conn = sqlite3.connect(db_path)
    query = """
    SELECT cd.*, wd.Population
    FROM covid_data cd
    LEFT JOIN worldometer_data wd
    ON cd.`Country.Region` = wd.`Country.Region`
    """
    df = pd.read_sql(query, conn)
    conn.close()

    countries = df['Country.Region'].unique()

    parameter_list = []
    for country in countries:
        df_country = df[df['Country.Region'] == country].copy()
        if len(df_country) < 10:  # Skip very small datasets to avoid errors
            continue
        params = estimate_parameters_for_country(df_country, country)
        if params:  # Ensure we have valid data
            parameter_list.append(params)

    df_params = pd.DataFrame(parameter_list)
    return df_params

def get_highest_parameters(df_params):
    highest_beta = df_params.loc[df_params['Beta'].idxmax()]
    highest_mu = df_params.loc[df_params['Mu'].idxmax()]
    highest_r0 = df_params.loc[df_params['R0'].idxmax()]
    return highest_beta, highest_r0, highest_mu
   
def display_dynamic_sir_insights(df_params):
    highest_beta, highest_r0, highest_mu = get_highest_parameters(df_params)

    st.markdown(f"""
    - **β**: In **{highest_beta['Country']}**, an infected individual, on average, affects **{highest_beta['Beta']:.2f}** susceptible individuals — highest estimated transmission rate.
    - **γ**: On average, individuals stay infected for **4.5 days** (assumed fixed average based on WHO estimates).
    - **μ**: In **{highest_mu['Country']}**, the estimated mortality rate of the virus is **{highest_mu['Mu']:.2%}** — highest estimated death rate among countries.
    """)

def plot_top5_death_rate_chart(top5_death_rate):
    fig = px.bar(
        top5_death_rate,
        x='DeathRate',
        y='Country.Region',
        orientation='h',
        color='DeathRate',
        color_continuous_scale='Reds',
        title="Top 5 Countries by Death Rate",
        labels={'Country.Region': 'Country', 'DeathRate': 'Death Rate (%)'},
    )
    fig.update_layout(yaxis={'categoryorder':'total ascending'})  # Sort bars properly
    st.plotly_chart(fig, use_container_width=True)



# -------- HOME PAGE ANALYTICS -------- #

def home_page_analytics(db_path="covid_database.db"):
    conn = sqlite3.connect(db_path)

    # Title for Home Page
    st.title("Global COVID-19 Insights and Epidemiological Analysis")
    st.markdown("""
    Welcome to the COVID-19 Data Analytics Dashboard — providing key insights on global pandemic trends, 
    mortality rates, and advanced epidemiological parameters (SIR-D Model) estimated from real-world data.
    """)

    # --- Fetch Top 5 Death Rate Countries for Graph --- #
    query_top5_death = """
    SELECT `Country.Region`, SUM(Deaths)*1.0 / SUM(Confirmed) AS DeathRate
    FROM covid_data
    WHERE Confirmed > 0
    GROUP BY `Country.Region`
    ORDER BY DeathRate DESC
    LIMIT 5;
    """
    top5_death_rate = pd.read_sql_query(query_top5_death, conn)
    top5_death_rate['DeathRate'] = top5_death_rate['DeathRate'] * 100  # Percentage

    # --- Estimate SIR parameters for all countries --- #
    query_params = """
    SELECT cd.*, wd.Population
    FROM covid_data cd
    LEFT JOIN worldometer_data wd
    ON cd.`Country.Region` = wd.`Country.Region`
    """
    df = pd.read_sql(query_params, conn)
    conn.close()  # Close connection once!

    # --- Estimate parameters ---
    countries = df['Country.Region'].unique()
    parameter_list = []

    for country in countries:
        df_country = df[df['Country.Region'] == country].copy()
        if len(df_country) < 10:
            continue  # Skip small datasets
        params = estimate_parameters_for_country(df_country, country)
        if params:
            parameter_list.append(params)

    df_params = pd.DataFrame(parameter_list)

    # --- Layout: Graph and Dynamic Insights --- #
    col1, col2 = st.columns(2)

    # Left side: Graph
    with col1:
        st.subheader("⚰️ Top 5 Countries by Death Rate")
        fig = px.bar(
            top5_death_rate,
            x='DeathRate',
            y='Country.Region',
            orientation='h',
            color='DeathRate',
            color_continuous_scale='Reds',
            title="Top 5 Countries by Death Rate",
            labels={'Country.Region': 'Country', 'DeathRate': 'Death Rate (%)'},
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})  # Sort bars properly
        st.plotly_chart(fig, use_container_width=True, key="top5_death_chart")

    # Right side: Dynamic SIR-D model insights
    with col2:
        st.subheader("🧬 SIR-D Parameter Insights (Real Data)")
        display_dynamic_sir_insights(df_params)

# -------- SIDEBAR NAVIGATION -------- #

st.sidebar.title("📊 COVID-19 Dashboard")
menu = st.sidebar.radio("Navigate", ["Home", "Global Data", "Region Data", "Country Data"])

# Filters for region and country view
region_list = fetch_region_list()
country_list = fetch_country_list()

selected_region = st.sidebar.selectbox("Select Region", region_list) if menu == "Region Data" else None
selected_country = st.sidebar.selectbox("Select Country", country_list) if menu == "Country Data" else None
start_date = st.sidebar.date_input("Start Date", datetime(2020, 1, 22))
end_date = st.sidebar.date_input("End Date", datetime(2020, 7, 27))


# -------- MAIN PAGE CONTENT -------- #

if menu == "Home":
    home_page_analytics()

elif menu == "Global Data":
    st.title("🌍 Global COVID-19 Data")
    df = fetch_global_data()
    df = filter_data_by_date(df, start_date, end_date)
    st.line_chart(df.set_index('Date')[['Active', 'Recovered', 'Deaths']])

elif menu == "Region Data" and selected_region:
    st.title(f"🌍 COVID-19 Data for {selected_region}")
    df = fetch_region_data(selected_region)
    df = filter_data_by_date(df, start_date, end_date)
    st.line_chart(df.set_index('Date')[['Active', 'Recovered', 'Deaths']])

elif menu == "Country Data" and selected_country:
    st.title(f"🌍 COVID-19 Data for {selected_country}")
    df = fetch_country_data(selected_country)
    df = filter_data_by_date(df, start_date, end_date)
    st.line_chart(df.set_index('Date')[['Active', 'Recovered', 'Deaths']])

# ----------------- Footer --------------------------

st.markdown("""
---
Made by Team 🚀 | COVID-19 Data Analytics Dashboard  
*For educational and research purposes only*
""")
