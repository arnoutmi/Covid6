import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import statsmodels as sm
import plotly.graph_objects as go


# -------- PAGE CONFIG -------- #
st.set_page_config(page_title="COVID-19 Data Dashboard", layout="wide")

st.markdown(
    """
    <style>
    /* Gradient fade sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(to bottom, #d9f8d9, #f5fff5);
    }
    </style>
    """,
    unsafe_allow_html=True
)
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

def fetch_us_county_list(db_path="covid_database.db"):
    """
    Fetches the list of unique USA counties from the usa_county_wise table.
    """
    import sqlite3
    import pandas as pd

    conn = sqlite3.connect(db_path)
    query = "SELECT DISTINCT Province_State FROM usa_county_wise ORDER BY Province_State"
    county_list = pd.read_sql_query(query, conn)["Province_State"].tolist()
    conn.close()

    return county_list

def fetch_country_data(country, db_path="covid_database.db"):
    conn = sqlite3.connect(db_path)
    query = """
    SELECT cd.Date, cd.Confirmed, cd.Active, cd.Recovered, cd.Deaths, wd.Population
    FROM covid_data cd
    LEFT JOIN worldometer_data wd
    ON cd.`Country.Region` = wd.`Country.Region`
    WHERE cd.`Country.Region` = ?
    ORDER BY cd.Date
    """
    df_country = pd.read_sql(query, conn, params=(country,))
    conn.close()
    df_country["Date"] = pd.to_datetime(df_country["Date"])
    return df_country

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

def fetch_global_data_filtered(start_date, end_date, db_path="covid_database.db"):
    conn = sqlite3.connect(db_path)
    query = """
        SELECT Date, SUM(Confirmed) AS Confirmed, SUM(Active) AS Active, 
               SUM(Recovered) AS Recovered, SUM(Deaths) AS Deaths
        FROM covid_data
        GROUP BY Date
        ORDER BY Date;
    """
    df = pd.read_sql(query, conn)
    conn.close()
    df["Date"] = pd.to_datetime(df["Date"])
    return df[(df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))]

def fetch_top_us_counties(db_path="covid_database.db"):
    import sqlite3
    import pandas as pd

    conn = sqlite3.connect(db_path)

    # Top 5 Counties by Deaths (latest date)
    deaths_query = """
        SELECT Province_State AS County, SUM(Deaths) AS Total_Deaths
        FROM usa_county_wise
        GROUP BY Province_State
        ORDER BY Total_Deaths DESC
        LIMIT 5;
    """
    deaths_df = pd.read_sql_query(deaths_query, conn)

    # Top 5 Counties by Confirmed Cases (latest date)
    cases_query = """
        SELECT Province_State AS County, SUM(Confirmed) AS Total_Cases
        FROM usa_county_wise
        GROUP BY Province_State
        ORDER BY Total_Cases DESC
        LIMIT 5;
    """
    cases_df = pd.read_sql_query(cases_query, conn)

    conn.close()
    return deaths_df, cases_df

def calculate_global_metrics(df):
    df = df.sort_values('Date')
    if df.empty:
        return 0, 0, 0, 0, 0.0
    confirmed, deaths, recovered, active = df.iloc[-1][['Confirmed', 'Deaths', 'Recovered', 'Active']] - df.iloc[0][['Confirmed', 'Deaths', 'Recovered', 'Active']]
    mortality_rate = (deaths / confirmed * 100) if confirmed > 0 else 0.0
    return confirmed, deaths, recovered, active, mortality_rate

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
    
def estimate_global_parameters_from_country_avg(df_params):
    """
    Compute global average of epidemiological parameters based on country estimates.
    """
    global_avg = {
        "Beta": df_params['Beta'].mean(),
        "Mu": df_params['Mu'].mean(),
        "R0": df_params['R0'].mean(),
        "Gamma": 1 / 4.5  # Fixed assumption
    }
    return global_avg

def estimate_parameters_for_region(region, db_path="covid_database.db"):
    """
    Estimates average β, μ, R₀ for a region, based on countries in that region.
    """
    conn = sqlite3.connect(db_path)
    query = """
    SELECT cd.*, wd.Population
    FROM covid_data cd
    LEFT JOIN worldometer_data wd
    ON cd.`Country.Region` = wd.`Country.Region`
    WHERE cd.Region = ?
    """
    df = pd.read_sql_query(query, conn, params=(region,))
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

    df_params = pd.DataFrame(parameter_list)

    if not df_params.empty:
        return {
            "Beta": df_params['Beta'].mean(),
            "Mu": df_params['Mu'].mean(),
            "R0": df_params['R0'].mean()
        }
    else:
        return {"Beta": 0, "Mu": 0, "R0": 0}

def fetch_country_wise_data(db_path="covid_database.db"):
    conn = sqlite3.connect(db_path)
    query = """
    SELECT cd.`Country.Region`, cd.Confirmed, cd.Active, cd.Recovered, cd.Deaths
    FROM covid_data cd
    INNER JOIN (
        SELECT `Country.Region`, MAX(Date) as max_date
        FROM covid_data
        GROUP BY `Country.Region`
    ) latest
    ON cd.`Country.Region` = latest.`Country.Region` AND cd.Date = latest.max_date
"""

    df_country = pd.read_sql_query(query, conn)
    conn.close()
    return df_country

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

def fetch_top_us_counties_by_latest(start_date, end_date, db_path="covid_database.db"):
    conn = sqlite3.connect(db_path)

    # Deaths based on latest available data within the range
    deaths_query = f"""
        SELECT Province_State AS County, MAX(Deaths) AS Total_Deaths
        FROM usa_county_wise
        WHERE Date <= '{end_date}'
        GROUP BY Province_State
        ORDER BY Total_Deaths DESC
        LIMIT 5;
    """
    deaths_df = pd.read_sql_query(deaths_query, conn)

    # Cases based on latest available data within the range
    cases_query = f"""
        SELECT Province_State AS County, MAX(Confirmed) AS Total_Cases
        FROM usa_county_wise
        WHERE Date <= '{end_date}'
        GROUP BY Province_State
        ORDER BY Total_Cases DESC
        LIMIT 5;
    """
    cases_df = pd.read_sql_query(cases_query, conn)

    conn.close()
    return deaths_df, cases_df

def plot_global_active_cases_per_population(db_path="covid_database.db"):
    conn = sqlite3.connect(db_path)
    query_global = """
    SELECT "Country.Region" as country, ActiveCases, Population
    FROM worldometer_data
    WHERE Population > 0 AND ActiveCases IS NOT NULL;
    """
    df_global = pd.read_sql_query(query_global, conn)
    conn.close()

    df_global['active_cases_per_population'] = df_global['ActiveCases'] / df_global['Population']

    fig = px.choropleth(
        df_global,
        locations="country",
        locationmode="country names",
        color="active_cases_per_population",
        hover_name="country",
        color_continuous_scale="RdYlGn_r",  # Green to Red (reversed for good/bad)
        title="Active COVID-19 Cases per Population (Global)",
        scope='world'
    )
    return fig

def plot_case_distribution_pie(total_active, total_recovered, total_deaths):
    pie_data = pd.DataFrame({
        'Category': ['Active', 'Recovered', 'Deaths'],
        'Count': [total_active, total_recovered, total_deaths]
    })
    fig = px.pie(
        pie_data,
        names='Category',
        values='Count',
        title='Distribution of Active, Recovered, and Deaths',
        color_discrete_sequence=px.colors.qualitative.Safe,
        width=400,  # Smaller size for better fit
        height=300
    )
    return fig

def plot_global_case_distribution_over_time(start_date, end_date, db_path="covid_database.db"):
    """
    Plots a stacked-like line chart showing the trend of Active, Recovered, and Deaths globally over time.
    Dynamically responds to date range selection.
    """

    # Step 1: Fetch and aggregate data
    conn = sqlite3.connect(db_path)
    query = """
        SELECT Date, 
               SUM(Active) AS Total_Active, 
               SUM(Recovered) AS Total_Recovered, 
               SUM(Deaths) AS Total_Deaths
        FROM covid_data
        GROUP BY Date
        ORDER BY Date;
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    # Step 2: Convert to datetime and filter by selected date range
    df["Date"] = pd.to_datetime(df["Date"])
    df_filtered = df[(df["Date"] >= pd.to_datetime(start_date)) & (df["Date"] <= pd.to_datetime(end_date))]

    # Step 3: Plot in "area-like" style (stacked feeling)
    fig = px.area(
        df_filtered,
        x='Date',
        y=['Total_Active', 'Total_Recovered', 'Total_Deaths'],  # Multiple categories
        title='Global Case Distribution Over Time',
        labels={"value": "Number of Cases", "variable": "Case Type", "Date": "Date"},
    )

    # Step 4: Customize size and aesthetics (to keep it clean)
    fig.update_layout(
        height=300,  # Control height (smaller as you wanted)
        width=600,   # Optional width
        legend_title="Case Type",
        margin=dict(l=20, r=20, t=50, b=20),  # Reduce margin for compactness
    )

    return fig

def plot_region_case_distribution_over_time(df_filtered, region_name):
    """
    Generates a stacked-like area chart for a specific region showing 
    Active, Recovered, and Deaths cases over time.

    Args:
        df_filtered (DataFrame): Region data filtered by date.
        region_name (str): Name of the selected region.
    Returns:
        plotly.graph_objs._figure.Figure: The generated Plotly area chart.
    """

    # Step 1: Prepare data (rename for consistency with global)
    df_filtered_for_graph = df_filtered.rename(columns={
        'Active': 'Total_Active',
        'Recovered': 'Total_Recovered',
        'Deaths': 'Total_Deaths'
    })

    # Step 2: Generate area plot
    fig = px.area(
        df_filtered_for_graph,
        x='Date',
        y=['Total_Active', 'Total_Recovered', 'Total_Deaths'],
        labels={"value": "Number of Cases", "variable": "Case Type", "Date": "Date"},
        color_discrete_map={
            'Total_Active': '#1f77b4',     # Blue (Active)
            'Total_Recovered': '#aec7e8',  # Light Blue (Recovered)
            'Total_Deaths': '#d62728'      # Red (Deaths)
        }
    )

    # Step 3: Update layout for visual consistency
    fig.update_layout(
    title="Region Case Distribution Over Time",  # Just the title as a string
    title_x=0.0,          # Horizontal position (0 for left)
    title_xanchor='left', # Anchor to the left
    title_y=0.95,         # Vertical position near top
    title_yanchor='top',  # Anchor to the top
    title_font=dict(size=16),  # Set font size
    height=300,  # Consistent height
    width=600,   # Optional width to match global
    legend_title="Case Type",
    margin=dict(l=20, r=20, t=50, b=20),  # Compact margins
)


    return fig

def plot_deaths_per_region_filtered_by_user_dates(start_date, end_date, db_path="covid_database.db"):
    """
    Compute total deaths per region dynamically based on selected start and end date.
    Only uses covid_data and sums deaths properly. Vertical bar chart with random colors.
    """

    # Connect and read data
    conn = sqlite3.connect(db_path)
    query = """
    SELECT Date, Region, SUM(Deaths) as Deaths
    FROM covid_data
    WHERE Region IS NOT NULL
    GROUP BY Date, Region
    ORDER BY Date
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    # Convert Date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Filter for end date
    df = df[(df['Date'] <= pd.to_datetime(end_date))]

    # Get deaths at start_date and end_date for each region
    deaths_start_df = df[df['Date'] <= pd.to_datetime(start_date)].groupby('Region')['Deaths'].max().reset_index()
    deaths_end_df = df[df['Date'] <= pd.to_datetime(end_date)].groupby('Region')['Deaths'].max().reset_index()

    # Merge start and end
    merged = pd.merge(deaths_end_df, deaths_start_df, on='Region', how='left', suffixes=('_end', '_start'))
    merged['Deaths_start'] = merged['Deaths_start'].fillna(0)  # Fill NaN with 0

    # Calculate total deaths within the period
    merged['Total_Deaths'] = merged['Deaths_end'] - merged['Deaths_start']
    merged = merged[merged['Total_Deaths'] > 0]  # Keep only regions with actual deaths

    # Sort by total deaths ascending (smallest to biggest)
    merged = merged.sort_values(by='Total_Deaths', ascending=True)

    # Plot vertical bar chart (from bottom to top)
    fig = px.bar(
        merged,
        x='Region',
        y='Total_Deaths',
        color='Region',  # Assign color per region
        color_discrete_sequence=px.colors.qualitative.Set3,  # Random distinct colors
        labels={'Total_Deaths': 'Total Deaths', 'Region': 'Region'},
        title='Deaths per Region (Selected Date Range)',
        height=400,  # You can adjust this
        width=600   # Optional width adjustment
    )

    fig.update_layout(
        showlegend=False,  # No need for legend since x-axis already labels regions
        xaxis_title='Region',
        yaxis_title='Total Deaths',
        xaxis={'categoryorder': 'total ascending'}  # Order categories from smallest to largest
    )

    return fig

def plot_top5_death_rate_chart(region, db_path="covid_database.db"):
    """
    Plots a bar chart of the top 5 countries by death rate within a specified region.
    """
    conn = sqlite3.connect(db_path)
    query_top5_death_rate = """
    SELECT cd.`Country.Region`, (cd.Deaths * 1.0 / cd.Confirmed) AS DeathRate
    FROM covid_data cd
    INNER JOIN (
    SELECT `Country.Region`, MAX(Date) as max_date
    FROM covid_data
    WHERE Confirmed > 0 AND Region = ?
    GROUP BY `Country.Region`
    ) latest
    ON cd.`Country.Region` = latest.`Country.Region` AND cd.Date = latest.max_date
    WHERE cd.Confirmed > 0
    ORDER BY DeathRate DESC
    LIMIT 5;"""

    top5_death_rate = pd.read_sql_query(query_top5_death_rate, conn, params=(region,))
    top5_death_rate['DeathRate'] = top5_death_rate['DeathRate'] * 100  # Convert to percentage
    conn.close()

    fig = px.bar(
        top5_death_rate,
        x='DeathRate',
        y='Country.Region',
        orientation='h',
        color='DeathRate',
        color_continuous_scale='Reds',
        labels={'Country.Region': 'Country', 'DeathRate': 'Death Rate (%)'},
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})  # Sort bars properly
    st.plotly_chart(fig, use_container_width=True)

def plot_top5_deaths_bar_region(selected_region, start_date, end_date, db_path="covid_database.db"):
    """
    Compute total deaths per country in a region dynamically based on selected start and end date.
    Vertical bar chart styled like global deaths per region.
    """
    conn = sqlite3.connect(db_path)
    query = """
    SELECT Date, `Country.Region`, Region, SUM(Deaths) as Deaths
    FROM covid_data
    WHERE Region = ?
    GROUP BY Date, `Country.Region`
    ORDER BY Date
    """
    df = pd.read_sql_query(query, conn, params=(selected_region,))
    conn.close()

    df['Date'] = pd.to_datetime(df['Date'])

    # Filter by end date
    df = df[(df['Date'] <= pd.to_datetime(end_date))]

    # Compute deaths at start and end dates
    deaths_start_df = df[df['Date'] <= pd.to_datetime(start_date)].groupby('Country.Region')['Deaths'].max().reset_index()
    deaths_end_df = df[df['Date'] <= pd.to_datetime(end_date)].groupby('Country.Region')['Deaths'].max().reset_index()

    # Merge and calculate
    merged = pd.merge(deaths_end_df, deaths_start_df, on='Country.Region', how='left', suffixes=('_end', '_start'))
    merged['Deaths_start'] = merged['Deaths_start'].fillna(0)
    merged['Total_Deaths'] = merged['Deaths_end'] - merged['Deaths_start']
    merged = merged[merged['Total_Deaths'] > 0]

    # Sort and get Top 5
    merged = merged.sort_values(by='Total_Deaths', ascending=False).head(5)

    # Plot - SAME style as global
    fig = px.bar(
        merged,
        x='Country.Region',
        y='Total_Deaths',
        color='Country.Region',
        color_discrete_sequence=px.colors.qualitative.Set3,
        labels={'Total_Deaths': 'Total Deaths', 'Country.Region': 'Country'},
        title='Top 5 Countries by Deaths (Region)'
    )

    fig.update_layout(
        showlegend=False,
        xaxis_title='Country',
        yaxis_title='Total Deaths',
        height=300,
        width=600,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis={'categoryorder': 'total descending'}
    )

    return fig

def plot_case_distribution_pie_chart(df, region, ax):
    """
    Plots a pie chart showing the distribution of active cases, recovered cases, and deaths for a specific region.
    """
    # Calculate the total numbers
    total_active = df["Active"].sum()
    total_recovered = df["Recovered"].sum()
    total_deaths = df["Deaths"].sum()

    # Data for the pie chart
    labels = ['Active Cases', 'Recovered Cases', 'Deaths']
    sizes = [total_active, total_recovered, total_deaths]
    colors = ['blue', 'green', 'red']

    # Plotting the pie chart
    ax.pie(sizes, labels=labels, colors=colors, startangle=90)
    
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

def plot_daily_new_cases(df):
    df['New Cases'] = df['Confirmed'].diff()
    fig = px.bar(
        df, x='Date', y='New Cases', color='New Cases', color_continuous_scale='Blues',
        labels={"New Cases": "Daily New Cases"},
        title= 'Daily New Cases'
    )
    fig.update_layout(
        height=300,
        width=600,
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=False
    )
    return fig

def plot_deaths_over_time(df):
    fig = px.area(
        df, 
        x='Date', 
        y='Deaths', 
        color_discrete_sequence=['red'],
        labels={"Deaths": "Daily Deaths"},
        title='Deaths Over Time'
    )
    fig.update_layout(
        height=300,
        width=600,
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=False
    )
    return fig

def plot_case_fatality_rate(df):
    df['CFR'] = (df['Deaths'] / df['Confirmed']) * 100
    fig = px.line(
        df, 
        x='Date', 
        y='CFR', 
        markers=True, 
        color_discrete_sequence=['black'],
        labels={"CFR": "Case Fatality Rate (%)"},
        title='Case Fatality Rate (CFR)'
    )
    fig.update_layout(
        height=300,
        width=600,
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=False
    )
    return fig

def plot_recovery_vs_death_rate(df):
    """Plots recovery rate vs. death rate over time."""
    st.subheader("💚 Recovery Rate vs ⚰️ Death Rate")
    df["Recovery Rate"] = (df["Recovered"] / df["Confirmed"]) * 100
    df["Death Rate"] = (df["Deaths"] / df["Confirmed"]) * 100
    fig = px.line(df, x="Date", y=["Recovery Rate", "Death Rate"], markers=True,
                  labels={"value": "Rate (%)", "variable": "Metric"})
    st.plotly_chart(fig, use_container_width=True)

def plot_daily_testing_positive_rate(df):
    """Plots percentage of positive tests if test data is available."""
    if "Tests" in df.columns and df["Tests"].notnull().sum() > 0:
        st.subheader("🧪 Test Positivity Rate Over Time")
        df["Positivity Rate"] = (df["Confirmed"] / df["Tests"]) * 100
        fig = px.line(df, x="Date", y="Positivity Rate", markers=True,
                      labels={"Positivity Rate": "Positive Test %"})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ No testing data available for this country.")

def plot_active_vs_recovered_cases(df):
    """Shows active vs recovered cases over time."""
    st.subheader("📊 Active vs. Recovered Cases")
    fig = px.area(df, x="Date", y=["Active", "Recovered"], 
                  labels={"value": "Number of Cases", "variable": "Case Type"})
    st.plotly_chart(fig, use_container_width=True)

def display_peak_case_and_death_days(df):
    """Displays the day with the highest new cases and deaths."""
    df["New Cases"] = df["Confirmed"].diff()
    df["New Deaths"] = df["Deaths"].diff()
    
    peak_cases_date = df.loc[df["New Cases"].idxmax(), "Date"]
    peak_cases = df["New Cases"].max()

    peak_deaths_date = df.loc[df["New Deaths"].idxmax(), "Date"]
    peak_deaths = df["New Deaths"].max()

    st.markdown(f"""
    🏆 **Peak New Cases:** {peak_cases:,} on **{peak_cases_date.date()}**  
    ⚰️ **Peak Deaths:** {peak_deaths:,} on **{peak_deaths_date.date()}**  
    """)

def plot_moving_average_trend(df, window=7):
    """Plots a moving average of daily new cases to smooth fluctuations."""
    st.subheader(f"📉 {window}-Day Moving Average of New Cases")
    df["New Cases"] = df["Confirmed"].diff()
    df["Moving Avg"] = df["New Cases"].rolling(window=window).mean()
    
    fig = px.line(df, x="Date", y="Moving Avg", markers=True, color_discrete_sequence=['blue'],
                  labels={"Moving Avg": f"{window}-Day Avg Cases"})
    st.plotly_chart(fig, use_container_width=True)

def plot_us_county_geographic_heatmap(df):
    """
    Creates a geographic heatmap showing COVID-19 Confirmed cases and Deaths by US counties.
    """
    # Get the latest available date
    df["Date"] = pd.to_datetime(df["Date"])
    latest_date = df["Date"].max()
    df_latest = df[df["Date"] == latest_date]

    # Create the scatter_geo plot
    fig = px.scatter_geo(
    df_latest,
    lat="Lat",
    lon="Long_",
    size="Confirmed",
    color="Deaths",
    hover_name="Combined_Key",
    scope='usa',
    color_continuous_scale="viridis",  # 🔥 Brighter contrast
    title=f"COVID-19 Geographic Distribution of US Counties (as of {latest_date.date()})",
)

    fig.update_layout(height=600)

    # Streamlit plot
    st.plotly_chart(fig, use_container_width=True)

def plot_death_vs_population_by_region(db_path="covid_database.db"):
    """
    Scatter plot comparing total deaths (latest date) vs population for each country.
    Colors indicate the region (from covid_data).
    """
    import sqlite3
    import pandas as pd
    import plotly.express as px
    import streamlit as st

    conn = sqlite3.connect(db_path)

    # Get latest date in covid_data
    latest_date_query = "SELECT MAX(Date) as latest_date FROM covid_data"
    latest_date = pd.read_sql_query(latest_date_query, conn).iloc[0]["latest_date"]

    # Fetch total deaths on latest date per country along with their region
    deaths_query = f"""
        SELECT `Country.Region`, SUM(Deaths) as Total_Deaths, Region
        FROM covid_data
        WHERE Date = '{latest_date}'
        GROUP BY `Country.Region`, Region
    """
    deaths_df = pd.read_sql_query(deaths_query, conn)

    # Fetch population per country
    pop_query = """
        SELECT `Country.Region`, Population
        FROM worldometer_data
        WHERE Population IS NOT NULL
    """
    pop_df = pd.read_sql_query(pop_query, conn)
    conn.close()

    # Merge population and deaths/region
    df = pd.merge(deaths_df, pop_df, on="Country.Region", how="inner")
    df = df[df["Total_Deaths"] > 0]  # Optional: Filter countries with no deaths

    # Color mapping for regions
    region_colors = {
    "Europe": "#1f77b4",               # Blue
    "Americas": "#d62728",             # Bright Red
    "Africa": "#2ca02c",               # Green
    "Eastern Mediterranean": "#FFD700",  # Yellow (Gold)
    "Western Pacific": "#ff7f0e",      # Orange
    "South-East Asia": "#e377c2"       # Pink
}

    # Plot the scatter
    fig = px.scatter(
        df,
        x="Population",
        y="Total_Deaths",
        color="Region",
        log_x=True,
        log_y=True, 
        color_discrete_map=region_colors,
        hover_name="Country.Region",
        labels={"Population": "Population", "Total_Deaths": "Total Deaths"},
    )
    
    
    fig.update_traces(marker=dict(size=10))

    min_x = df["Population"].min()
    max_x = df["Population"].max()
    min_y = df["Total_Deaths"].min()
    max_y = df["Total_Deaths"].max()

# Adjust the slope based on data ratio (like y = x)
    fig.add_shape(
    type="line",
    x0=min_x,
    y0=min_y,
    x1=max_x,
    y1=max_y,
    line=dict(color="black", width=2, dash="dash")
)

    fig.update_layout(
        title="Total Deaths vs Population by Region",
        xaxis_title="Population",
        yaxis_title="Total Deaths",
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)

def plot_confirmed_vs_population_by_region(db_path="covid_database.db"):
    """
    Scatter plot comparing total confirmed cases (latest date) vs population for each country.
    Colors indicate the region (from covid_data).
    """
    import sqlite3
    import pandas as pd
    import plotly.express as px
    import streamlit as st

    conn = sqlite3.connect(db_path)

    # Get latest date in covid_data
    latest_date_query = "SELECT MAX(Date) as latest_date FROM covid_data"
    latest_date = pd.read_sql_query(latest_date_query, conn).iloc[0]["latest_date"]

    # Fetch total confirmed cases on latest date per country along with their region
    confirmed_query = f"""
        SELECT `Country.Region`, SUM(Confirmed) as Total_Confirmed, Region
        FROM covid_data
        WHERE Date = '{latest_date}'
        GROUP BY `Country.Region`, Region
    """
    confirmed_df = pd.read_sql_query(confirmed_query, conn)

    # Fetch population per country
    pop_query = """
        SELECT `Country.Region`, Population
        FROM worldometer_data
        WHERE Population IS NOT NULL
    """
    pop_df = pd.read_sql_query(pop_query, conn)
    conn.close()

    # Merge population and confirmed/region
    df = pd.merge(confirmed_df, pop_df, on="Country.Region", how="inner")
    df = df[df["Total_Confirmed"] > 0]  # Filter countries with no confirmed cases

    # Color mapping for regions (with Eastern Mediterranean as yellow)
    region_colors = {
        "Europe": "#1f77b4",                 # Blue
        "Americas": "#d62728",               # Red
        "Africa": "#2ca02c",                 # Green
        "Eastern Mediterranean": "#FFD700",  # Yellow (Gold)
        "Western Pacific": "#ff7f0e",        # Orange
        "South-East Asia": "#e377c2"         # Pink
    }

    # Plot the scatter
    fig = px.scatter(
        df,
        x="Population",
        y="Total_Confirmed",
        color="Region",
        log_x=True,
        log_y=True,
        color_discrete_map=region_colors,
        hover_name="Country.Region",
        labels={"Population": "Population", "Total_Confirmed": "Total Confirmed Cases"},
    )
    fig.update_traces(marker=dict(size=10))

    # Add diagonal reference line
    fig.add_shape(
        type="line",
        x0=df["Population"].min(),
        y0=df["Total_Confirmed"].min(),
        x1=df["Population"].max(),
        y1=df["Total_Confirmed"].max(),
        line=dict(color="black", width=2, dash="dash")
    )

    fig.update_layout(
        title="Total Confirmed Cases vs Population by Region",
        xaxis_title="Population",
        yaxis_title="Total Confirmed Cases",
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)

def country_dashboard_page(selected_country, start_date, end_date, db_path="covid_database.db"):
    # Fetch country data
    df_country = fetch_country_data(selected_country)
    df_filtered = filter_data_by_date(df_country, start_date, end_date)

    # Compute country metrics
    total_confirmed = df_filtered["Confirmed"].iloc[-1] - df_filtered["Confirmed"].iloc[0] if not df_filtered.empty else 0
    total_deaths = df_filtered["Deaths"].iloc[-1] - df_filtered["Deaths"].iloc[0] if not df_filtered.empty else 0
    total_recovered = df_filtered["Recovered"].iloc[-1] - df_filtered["Recovered"].iloc[0] if not df_filtered.empty else 0
    total_active = df_filtered["Active"].iloc[-1] - df_filtered["Active"].iloc[0] if not df_filtered.empty else 0
    mortality_rate = (total_deaths / total_confirmed * 100) if total_confirmed > 0 else 0.0

    st.markdown("""
        <style>
        .box-yellow { background-color: #FFF9DB; padding: 15px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); margin-bottom: 15px; }
        .box-green { background-color: #E8F8F5; padding: 15px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); margin-bottom: 15px; }
        .box-blue { background-color: #E8F4FD; padding: 15px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); margin-bottom: 15px; }
        .box-rose { background-color: #FDE2E4; padding: 15px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); margin-bottom: 15px; }
        </style>
    """, unsafe_allow_html=True)


    # -------- DASHBOARD HEADER -------- #
    st.markdown(f"""
    <h4 style='text-align: center;'>🌍 COVID-19 Analytics Dashboard for {selected_country}</h4>
    <h6 style='text-align: center;'>Analysis Period: <b>{start_date.strftime('%Y-%m-%d')}</b> to <b>{end_date.strftime('%Y-%m-%d')}</b></h6>
    <hr style='border:1px solid #e0e0e0;'>
    """, unsafe_allow_html=True)

    country_params = estimate_parameters_for_country(df_country, selected_country)

    # --------- FIRST ROW: METRICS, AREA CHART, PIE CHART --------- #
    col1, col2, col3 = st.columns([2, 4, 3])

    # Box 1: Metrics
    with col1:
        st.markdown(f"""
        <div class="box-yellow">
            <h4>📊 {selected_country} Metrics</h4>
            <p><b>Total Confirmed Cases:</b> {total_confirmed:,}</p>
            <p><b>Total Deaths:</b> {total_deaths:,}</p>
            <p><b>Mortality Rate (%):</b> {mortality_rate:.2f}%</p>
            <p><b>β (Transmission Rate):</b> {country_params['Beta']:.2f}</p>
            <p><b>R₀ (Reproduction Number):</b> {country_params['R0']:.2f}</p>
        </div>
        """, unsafe_allow_html=True)

    # Box 2: Area Chart (Stacked)
    with col2:
        st.markdown("""
        <div class="box-green"></div>
        """, unsafe_allow_html=True)

        fig_area = px.area(
            df_filtered,
            x="Date",
            y=["Active", "Recovered", "Deaths"],
            color_discrete_map={"Active": '#1f77b4', "Recovered": '#2ca02c', "Deaths": '#d62728'},
            labels={"value": "Number of Cases", "variable": "Case Type"},
            title=f"{selected_country} Case Distribution Over Time"
        )
        fig_area.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_area, use_container_width=True)

    # Box 3: Pie Chart
    with col3:
        st.markdown("""
        <div class="box-blue"></div>
        """, unsafe_allow_html=True)

        pie_chart = plot_case_distribution_pie(total_active, total_recovered, total_deaths)
        st.plotly_chart(pie_chart, use_container_width=True)

    col4, col5, col6 = st.columns([1, 1, 1])

    with col4:
        with st.container():
        # Small green box for aesthetic (no title inside)
            st.markdown("""
            <div class="box-green" style="padding: 15px; border-radius: 10px; 
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
            </div>
        """, unsafe_allow_html=True)
        daily_new_cases_fig = plot_daily_new_cases(df_filtered)
        st.plotly_chart(daily_new_cases_fig, use_container_width=True)


    with col5:
        st.markdown("""
            <div class="box-rose" style="padding: 15px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
            </div>        
        """, unsafe_allow_html=True)
        death_plot_fig = plot_deaths_over_time(df_filtered)
        st.plotly_chart(death_plot_fig, use_container_width=True)

    with col6:
        st.markdown("""
            <div class="box-yellow
            " style="padding: 15px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
            </div>        
        """, unsafe_allow_html=True)
        cfr_plot_fig = plot_case_fatality_rate(df_filtered)
        st.plotly_chart(cfr_plot_fig, use_container_width=True)
       


    # -------- FOOTER -------- #
    st.markdown("---")
    st.markdown("*For educational and research purposes only*")

def global_dashboard_page(start_date, end_date, db_path="covid_database.db"):
    # ---------------- FETCH GLOBAL DATA ---------------- #
    df_global = fetch_global_data_filtered(start_date, end_date, db_path)
    df_params = estimate_parameters_all_countries()
    global_params = estimate_global_parameters_from_country_avg(df_params)
    total_confirmed, total_deaths, total_recovered, total_active, mortality_rate = calculate_global_metrics(df_global)

    st.markdown("""
        <style>
        .box-yellow { background-color: #FFF9DB; padding: 15px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); margin-bottom: 15px; }
        .box-green { background-color: #E8F8F5; padding: 15px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); margin-bottom: 15px; }
        .box-blue { background-color: #E8F4FD; padding: 15px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); margin-bottom: 15px; }
        .box-rose { background-color: #FDE2E4; padding: 15px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); margin-bottom: 15px; }
        </style>
    """, unsafe_allow_html=True)


    # ---------------- DASHBOARD TITLE ---------------- #
    st.markdown("""
        <h4 style='text-align: center; color: #333333; margin-bottom: 0;'>🌍 Global COVID-19 Analytics Dashboard</h4>
        <h6 style='text-align: center; color: #555555; margin-top: 0;'>Analysis: <b>{}</b> to <b>{}</b></h6>
        <hr style='border:1px solid #e0e0e0;'>

    """.format(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')), unsafe_allow_html=True)

    # --------- ROW 1: METRICS & PIE CHART --------- #
    col1, col2, col3 = st.columns([2, 4, 3])  # Adjusted to fit better

    # ---- Box 1: Confirmed & Deaths ---- #
    with col1:
        st.markdown(f"""
        <div class="box-yellow" style="padding: 15px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
            <h4>📊 Confirmed & Deaths</h4>
            <p><b>Total Confirmed Cases:</b> {total_confirmed:,}</p>
            <p><b>Total Deaths:</b> {total_deaths:,}</p>
            <p><b>Mortality Rate (%):</b> {mortality_rate:.2f}%</p>
            <b>β (Transmission Rate):</b> {global_params['Beta']:.2f}<p>
            <b>R₀ (Reproduction Number):</b> {global_params['R0']:.2f}</p>


        </div>
        """, unsafe_allow_html=True)

    # ---- Box 2: Global Parameters ---- #
    
    with col2:
        with st.container():
            st.markdown("""
            <div class="box-green" style="padding: 15px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
            </div>
            """, unsafe_allow_html=True)

            # Generate dynamic line graph for case distribution
            case_distribution_chart = plot_global_case_distribution_over_time(start_date=start_date, end_date=end_date, db_path=db_path)
            st.plotly_chart(case_distribution_chart, use_container_width=True)


    # ---- Box 3: Pie Chart ---- #
    with col3:
        st.markdown(f"""
        <div class="box-blue" style="padding: 15px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
        </div>
        """, unsafe_allow_html=True)
        pie_chart = plot_case_distribution_pie(total_active, total_recovered, total_deaths)
        st.plotly_chart(pie_chart, use_container_width=True)

    # --------- ROW 2: MAP + DEATH PER REGION --------- #
    col4, col5 = st.columns([4, 2])

    # ---- Box 4: Global Map ---- #
    with col4:
        st.markdown(f"""
        <div class="box-rose" style="padding: 15px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
        </div>
        """, unsafe_allow_html=True)
        world_map = plot_global_active_cases_per_population(db_path)
        st.plotly_chart(world_map, use_container_width=True)
    
    with col5:
        with st.container():
            st.markdown("""
                <div style='background-color: #fff3cd; padding: 15px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                </div>
            """, unsafe_allow_html=True)

        deaths_region_chart = plot_deaths_per_region_filtered_by_user_dates(start_date=start_date, end_date=end_date, db_path=db_path)
        st.plotly_chart(deaths_region_chart, use_container_width=True)

    # ---------------- FOOTER ---------------- #
    st.markdown("---")
    st.markdown("*For educational and research purposes only*")

def region_dashboard_page(selected_region, start_date, end_date, db_path="covid_database.db"):
    df_region = fetch_region_data(selected_region, db_path)
    df_filtered = filter_data_by_date(df_region, start_date, end_date)

    total_confirmed = df_filtered["Confirmed"].iloc[-1] - df_filtered["Confirmed"].iloc[0] if not df_filtered.empty else 0
    total_deaths = df_filtered["Deaths"].iloc[-1] - df_filtered["Deaths"].iloc[0] if not df_filtered.empty else 0
    total_recovered = df_filtered["Recovered"].iloc[-1] - df_filtered["Recovered"].iloc[0] if not df_filtered.empty else 0
    total_active = df_filtered["Active"].iloc[-1] - df_filtered["Active"].iloc[0] if not df_filtered.empty else 0
    mortality_rate = (total_deaths / total_confirmed * 100) if total_confirmed > 0 else 0.0

    st.markdown("""
        <style>
        .box-yellow { background-color: #FFF9DB; padding: 15px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); margin-bottom: 15px; }
        .box-green { background-color: #E8F8F5; padding: 15px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); margin-bottom: 15px; }
        .box-blue { background-color: #E8F4FD; padding: 15px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); margin-bottom: 15px; }
        .box-rose { background-color: #FDE2E4; padding: 15px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); margin-bottom: 15px; }
        </style>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <h4 style='text-align: center;'>🌎 COVID-19 Analytics Dashboard for {selected_region}</h4>
    <h6 style='text-align: center;'>Analysis Period: <b>{start_date.strftime('%Y-%m-%d')}</b> to <b>{end_date.strftime('%Y-%m-%d')}</b></h6>
    <hr style='border:1px solid #e0e0e0;'>
    """, unsafe_allow_html=True)

    region_params = estimate_parameters_for_region(selected_region, db_path=db_path)

    # ----- FIRST ROW ----- #
    col1, col2, col3 = st.columns([2, 4, 3])

    with col1:
        st.markdown(f"""
        <div class="box-yellow">
            <h4>📊 {selected_region} Metrics</h4>
            <p><b>Total Confirmed Cases:</b> {total_confirmed:,}</p>
            <p><b>Total Deaths:</b> {total_deaths:,}</p>
            <p><b>Mortality Rate (%):</b> {mortality_rate:.2f}%</p>
            <p><b>β (Transmission Rate):</b> {region_params['Beta']:.2f}</p>
            <p><b>R₀ (Reproduction Number):</b> {region_params['R0']:.2f}</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""<div class="box-green"></div>""", unsafe_allow_html=True)
        fig_region_area = plot_region_case_distribution_over_time(df_filtered, selected_region)
        st.plotly_chart(fig_region_area, use_container_width=True)

    with col3:
        st.markdown("""<div class="box-blue"></div>""", unsafe_allow_html=True)
        pie_chart = plot_case_distribution_pie(total_active, total_recovered, total_deaths)
        st.plotly_chart(pie_chart, use_container_width=True)

    # ----- SECOND ROW ----- #
    col4, col5, col6 = st.columns([1, 1, 1])

    with col4:
        st.markdown("""<div class="box-green"></div>""", unsafe_allow_html=True)
        st.plotly_chart(plot_daily_new_cases(df_filtered), use_container_width=True)

    with col5:
        st.markdown("""<div class="box-rose"></div>""", unsafe_allow_html=True)
        st.plotly_chart(plot_deaths_over_time(df_filtered), use_container_width=True)

    with col6:
        st.markdown("""<div class="box-yellow"></div>""", unsafe_allow_html=True)
        region_deaths_bar = plot_top5_deaths_bar_region(selected_region, start_date, end_date, db_path)
        st.plotly_chart(region_deaths_bar, use_container_width=True)


    # ----- FOOTER ----- #
    st.markdown("---")
    st.markdown("*For educational and research purposes only*")

def us_county_summary_dashboard(start_date, end_date, db_path="covid_database.db"):
    import streamlit as st
    import plotly.express as px

    deaths_df, cases_df = fetch_top_us_counties_by_latest(start_date, end_date)

    st.header("🇺🇸 US Counties with Highest COVID-19 Impact")

    col1, col2=  st.columns(2)

    with col1:
        st.subheader("💀 Top 5 Counties by Deaths")
        fig_deaths = px.bar(
            deaths_df,
            x="Total_Deaths",
            y="County",
            orientation='h',
            color="Total_Deaths",
            color_continuous_scale="Reds",
            labels={"Total_Deaths": "Total Deaths", "County": "County"},
            title="Top 5 Deadliest Counties (Latest)"
        )
        st.plotly_chart(fig_deaths, use_container_width=True)

    with col2:
        st.subheader("🦠 Top 5 Counties by Cases")
        fig_cases = px.bar(
            cases_df,
            x="Total_Cases",
            y="County",
            orientation='h',
            color="Total_Cases",
            color_continuous_scale="Blues",
            labels={"Total_Cases": "Total Cases", "County": "County"},
            title="Top 5 Counties with Most Cases (Latest)"
        )
        st.plotly_chart(fig_cases, use_container_width=True)
    
    st.markdown("---")
    st.subheader("🗺 COVID-19 Geographic Heatmap of US Counties")

    # Fetch data for heatmap
    conn = sqlite3.connect(db_path)
    geo_query = """
        SELECT Date, Lat, Long_, Confirmed, Deaths, Combined_Key
        FROM usa_county_wise
    """
    geo_df = pd.read_sql_query(geo_query, conn)
    conn.close()

    # Plot the geographic heatmap using your function
    plot_us_county_geographic_heatmap(geo_df)
    st.markdown("---")
    st.markdown("*Data visualized by the COVID-19 Analytics Dashboard*")

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

def user_controlled_analysis_page():
    st.markdown("""
        <h4 style='text-align: center;'>🧠 User-Controlled COVID-19 Analysis</h4>
        <hr style='border:1px solid #e0e0e0; margin-top:10px;'>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)  # Space for visual balance

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analysis_choice = st.selectbox(
            "Choose what you want to analyze:",
            ["Select Analysis Option...",
            "Compare Two Countries",
            "Death vs Population Scatter",
            "Confirmed vs Population Scatter"]
        )

    if analysis_choice == "Compare Two Countries":
        country_list = fetch_country_list()
        col_a, col_b = st.columns(2)

        with col_a:
            country1 = st.selectbox("Select First Country", country_list)
        with col_b:
            country2 = st.selectbox("Select Second Country", country_list, index=1)

        compare_option = st.selectbox("What do you want to compare?", [
            "Deaths",
            "Confirmed",
            "Recovered",
            "Daily New Cases",
            "Deaths per 100k population",
            "Recovered per 100k population",
            "Case Growth Rate Over Time",
            "Case Fatality Rate (CFR)"
        ])

        if st.button("Compare"):
            df1 = fetch_country_data(country1)
            df2 = fetch_country_data(country2)

            # Align dates
            df1 = df1.sort_values("Date").reset_index(drop=True)
            df2 = df2.sort_values("Date").reset_index(drop=True)

            # Prepare plot
            plot_df = pd.DataFrame()
            plot_df['Date'] = df1['Date']

            if compare_option == "Deaths":
                plot_df[country1] = df1["Deaths"]
                plot_df[country2] = df2["Deaths"]
                y_label = "Deaths"
            elif compare_option == "Confirmed":
                plot_df[country1] = df1["Confirmed"]
                plot_df[country2] = df2["Confirmed"]
                y_label = "Confirmed Cases"
            elif compare_option == "Recovered":
                plot_df[country1] = df1["Recovered"]
                plot_df[country2] = df2["Recovered"]
                y_label = "Recovered"
            elif compare_option == "Daily New Cases":
                plot_df[country1] = df1["Confirmed"].diff()
                plot_df[country2] = df2["Confirmed"].diff()
                y_label = "Daily New Cases"
            elif compare_option == "Deaths per 100k population":
                plot_df[country1] = (df1["Deaths"] / df1["Population"]) * 100000
                plot_df[country2] = (df2["Deaths"] / df2["Population"]) * 100000
                y_label = "Deaths per 100k"
            elif compare_option == "Recovered per 100k population":
                plot_df[country1] = (df1["Recovered"] / df1["Population"]) * 100000
                plot_df[country2] = (df2["Recovered"] / df2["Population"]) * 100000
                y_label = "Recovered per 100k"
            elif compare_option == "Case Growth Rate Over Time":
                plot_df[country1] = df1["Confirmed"].pct_change() * 100
                plot_df[country2] = df2["Confirmed"].pct_change() * 100
                y_label = "Growth Rate (%)"
            elif compare_option == "Case Fatality Rate (CFR)":
                plot_df[country1] = (df1["Deaths"] / df1["Confirmed"]) * 100
                plot_df[country2] = (df2["Deaths"] / df2["Confirmed"]) * 100
                y_label = "CFR (%)"

            # Plotly chart
            fig = px.line(
                plot_df,
                x="Date",
                y=[country1, country2],
                labels={"value": y_label, "variable": "Country"},
                title=f"Comparison of {compare_option} between {country1} and {country2}"
            )

            fig.update_layout(height=400, width=800, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)
    elif analysis_choice == "Death vs Population Scatter":
        st.subheader("🌍 Scatter Plot: Total Deaths vs Population by Region")
        plot_death_vs_population_by_region()
    
    elif analysis_choice == "Confirmed vs Population Scatter":
        st.subheader("🌍 Scatter Plot: Total Confirmed Cases vs Population by Region")
        plot_confirmed_vs_population_by_region()

# -------- SIDEBAR NAVIGATION -------- #

st.sidebar.title("📊 COVID-19 Dashboard")
menu = st.sidebar.radio("Navigate", ["Home", "Global Data", "Region Data", "Country Data", "USA County Data", "User-Controlled Analysis"])

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
    global_dashboard_page(start_date, end_date)


elif menu == "Region Data" and selected_region:
    region_dashboard_page(selected_region, start_date, end_date)

elif menu == "Country Data" and selected_country:
    country_dashboard_page(selected_country, start_date, end_date)

elif menu == "USA County Data":
    us_county_summary_dashboard(start_date, end_date)
elif menu == "User-Controlled Analysis":
    user_controlled_analysis_page()

# ----------------- Footer --------------------------

st.markdown("""
---
Made by Team 🚀 | COVID-19 Data Analytics Dashboard  
*For educational and research purposes only*
""")
