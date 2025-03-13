import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px

def fix_missing_values(csv_path, db_path="covid_database.db", method="interpolate"):
    """
    Loads the complete.csv dataset into an SQLite database.
    Groups data by Country, Region, and Date, fills missing values using the selected method,
    and stores the cleaned data in the database.

    Parameters:
    - csv_path: Path to the CSV file
    - db_path: Path to the SQLite database
    - method: Method for filling missing values ("ffill" for forward-fill, "interpolate" for interpolation)
    """
    df = pd.read_csv(csv_path)

    df["Date"] = pd.to_datetime(df["Date"])

    df.rename(columns={"WHO.Region": "Region"}, inplace=True)

    df = df[["Date", "Country.Region", "Region", "Confirmed", "Active", "Recovered", "Deaths"]]

    df = df.groupby(["Country.Region", "Region", "Date"]).sum().reset_index()

    df.drop_duplicates(inplace=True)  

    if method == "ffill":
        df[["Active", "Recovered", "Deaths"]] = df.groupby("Country.Region")[["Active", "Recovered", "Deaths"]].ffill()
    elif method == "interpolate":
        df[["Active", "Recovered", "Deaths"]] = (
            df.groupby("Country.Region")[["Active", "Recovered", "Deaths"]]
            .apply(lambda x: x.interpolate(method="linear"))
            .reset_index(drop=True) 
        )

    conn = sqlite3.connect(db_path)
    df.to_sql("covid_data", conn, if_exists="replace", index=False)
    conn.close()   

def fetch_country_data(country, db_path="covid_database.db"):
    """
    Fetches COVID-19 data for a specific country and calculates cumulative totals.
    """
    conn = sqlite3.connect(db_path)
    query = "SELECT Date, Confirmed, Active, Recovered, Deaths FROM covid_data WHERE `Country.Region` = ?"
    df_country = pd.read_sql(query, conn, params=(country,))
    conn.close()

    df_country["Date"] = pd.to_datetime(df_country["Date"])
    df_country.sort_values("Date", inplace=True)

    df_country["Total_Active"] = df_country["Active"].cumsum()
    df_country["Total_Recovered"] = df_country["Recovered"].cumsum()

    return df_country

def fetch_global_data(db_path="covid_database.db"):
    """
    Fetches global COVID-19 data by aggregating all country data for each date.
    Adds cumulative total columns for Active and Recovered cases.
    """
    conn = sqlite3.connect(db_path)
    query = """
        SELECT Date, 
               SUM(Active) AS Active, 
               SUM(Recovered) AS Recovered, 
               SUM(Deaths) AS Deaths
        FROM covid_data
        GROUP BY Date
        ORDER BY Date;
    """
    df_global = pd.read_sql(query, conn)
    conn.close()

    # Convert Date to datetime format
    df_global["Date"] = pd.to_datetime(df_global["Date"])

    # Add cumulative total columns
    df_global["Total_Active"] = df_global["Active"].cumsum()
    df_global["Total_Recovered"] = df_global["Recovered"].cumsum()

    return df_global

def fetch_region_data(region, db_path="covid_database.db"):
    """
    Fetches COVID-19 data for a specific region and calculates cumulative totals.
    """
    conn = sqlite3.connect(db_path)
    query = """
        SELECT Date, SUM(Active) AS Active, SUM(Recovered) AS Recovered, SUM(Deaths) AS Deaths
        FROM covid_data
        WHERE Region = ?
        GROUP BY Date
        ORDER BY Date;
    """
    df_region = pd.read_sql(query, conn, params=(region,))
    conn.close()

    df_region["Date"] = pd.to_datetime(df_region["Date"])

    # Add cumulative total columns
    df_region["Total_Active"] = df_region["Active"].cumsum()
    df_region["Total_Recovered"] = df_region["Recovered"].cumsum()

    return df_region


def fetch_population(country, db_path="covid_database.db"):
    conn = sqlite3.connect(db_path)
    query = "SELECT Population FROM worldometer_data WHERE `Country.Region` = ?"
    df = pd.read_sql(query, conn, params=(country,))
    conn.close()
    return df["Population"].iloc[0] if not df.empty else None


def covid_trends_country(df, country):
    """
    Plots the COVID-19 trends (Active, Recovered, Deaths) for a given country.
    """

    fig, axes = plt.subplots(3, 1, figsize=(15, 9))

    axes[0].plot(df["Date"], df["Active"], color="blue", label="Active Cases")
    axes[0].set_title(f"Daily Active COVID-19 Cases in {country}")

    fig, axes = plt.subplots(3, 1, figsize=(15, 9))

    axes[0].plot(df["Date"], df["Active"], color="blue", label="Active Cases")
    axes[0].set_title(f"Daily Active COVID-19 Cases in {country}")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Cases")
    axes[0].legend()

    axes[1].plot(df["Date"], df["Deaths"], color="red", label="Deaths")
    axes[1].set_title(f"Daily COVID-19 Deaths in {country}")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Deaths")
    axes[1].legend()

    axes[2].plot(df["Date"], df["Recovered"], color="green", label="Recoveries")
    axes[2].set_title(f"Daily Recoveries from COVID-19 in {country}")
    axes[2].set_xlabel("Date")
    axes[2].set_ylabel("Recoveries")
    axes[2].legend()

    plt.tight_layout()
    plt.show()

def covid_trends_region(df, region):
    """
    Plots the COVID-19 trends (Active, Recovered, Deaths) for a given region.
    """
    fig, axes = plt.subplots(3, 1, figsize=(15, 9))

    axes[0].plot(df["Date"], df["Active"], color="blue", label="Active Cases")
    axes[0].set_title(f"Daily Active COVID-19 Cases in {region}")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Cases")
    axes[0].legend()

    axes[1].plot(df["Date"], df["Deaths"], color="red", label="Deaths")
    axes[1].set_title(f"Daily COVID-19 Deaths in {region}")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Deaths")
    axes[1].legend()

    axes[2].plot(df["Date"], df["Recovered"], color="green", label="Recoveries")
    axes[2].set_title(f"Daily Recoveries from COVID-19 in {region}")
    axes[2].set_xlabel("Date")
    axes[2].set_ylabel("Recoveries")
    axes[2].legend()

    plt.tight_layout()
    plt.show()

def covid_trends_global(df):
    """
    Plots the COVID-19 trends (Active, Recovered, Deaths) globally.
    """
    fig, axes = plt.subplots(3, 1, figsize=(15, 9))

    axes[0].plot(df["Date"], df["Active"], color="blue", label="Active Cases")
    axes[0].set_title(f"Daily Active COVID-19 Cases Globally")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Cases")
    axes[0].legend()

    axes[1].plot(df["Date"], df["Deaths"], color="red", label="Deaths")

    
    axes[1].set_title(f"Daily COVID-19 Deaths Globally")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Deaths")
    axes[1].legend()

    axes[2].plot(df["Date"], df["Recovered"], color="green", label="Recoveries")

   

    axes[2].set_title(f"Daily Recoveries from COVID-19 Globally")

    axes[2].set_xlabel("Date")
    axes[2].set_ylabel("Recoveries")
    axes[2].legend()

    plt.tight_layout()
    plt.show()

def active_recovered_deaths_global(df):
    """
    Plots Active Cases, Deaths, and Recoveries globally on the same graph.
    """
    plt.figure(figsize=(15, 6))

    plt.plot(df["Date"], df["Total_Active"], color="blue", label="Total Active Cases")
    plt.plot(df["Date"], df["Total_Recovered"], color="green", label="Total Recovered Cases")
    plt.plot(df["Date"], df["Deaths"].cumsum(), color="red", label="Total Deaths")

    plt.title("Global COVID-19 Trends: Total Active Cases, Recovered, and Deaths")
    plt.xlabel("Date")
    plt.ylabel("Number of Cases")
    plt.legend()
    plt.grid(True)

    plt.show()

def active_recovered_deaths_country(df, country):
    """
    Plots Active Cases, Deaths, and Recoveries for a specific country on the same graph.
    """
    plt.figure(figsize=(15, 6))

    plt.plot(df["Date"], df["Total_Active"], color="blue", label="Total Active Cases")
    plt.plot(df["Date"], df["Total_Recovered"], color="green", label="Total Recovered Cases")
    plt.plot(df["Date"], df["Deaths"].cumsum(), color="red", label="Total Deaths")

    plt.title(f"COVID-19 Trends in {country}: Total Active Cases, Recovered, and Deaths")
    plt.xlabel("Date")
    plt.ylabel("Number of Cases")
    plt.legend()
    plt.grid(True)

    plt.show()

def active_recovered_deaths_region(df, region):
    """
    Plots Active Cases, Deaths, and Recoveries for a specific region on the same graph.
    """
    plt.figure(figsize=(15, 6))

    plt.plot(df["Date"], df["Total_Active"], color="blue", label="Total Active Cases")
    plt.plot(df["Date"], df["Total_Recovered"], color="green", label="Total Recovered Cases")
    plt.plot(df["Date"], df["Deaths"].cumsum(), color="red", label="Total Deaths")

    plt.title(f"COVID-19 Trends in {region}: Total Active Cases, Recovered, and Deaths")
    plt.xlabel("Date")
    plt.ylabel("Number of Cases")
    plt.legend()
    plt.grid(True)

    plt.show()

def plot_death_rate_by_continent(db_path="covid_database.db"):
    """
    Plots the COVID-19 death rate by continent using the correct column names.
    """
    conn = sqlite3.connect(db_path)
    query_deaths = """
    SELECT Continent, SUM(TotalDeaths) as total_deaths, SUM(Population) as population
    FROM worldometer_data
    GROUP BY Continent;
    """
    df_continents = pd.read_sql_query(query_deaths, conn)
    conn.close()

    # Calculate Death Rate (μ) per Continent
    df_continents['death_rate'] = df_continents['total_deaths'] / df_continents['population'] * 100

    # Create a bar plot to visualize the death rate for each continent
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Continent', y='death_rate', data=df_continents, palette='viridis')
    plt.title('COVID-19 Death Rate by Continent')
    plt.xlabel('Continent')
    plt.ylabel('Death Rate (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_deaths_by_us_county(db_path="covid_database.db"):
    """
    Fetches and plots the total COVID-19 deaths for each county in the USA.
    """
    conn = sqlite3.connect(db_path)
    query = """
    SELECT Province_State AS County, SUM(Deaths) AS Total_Deaths
    FROM usa_county_wise
    GROUP BY Province_State
    ORDER BY Total_Deaths DESC
    LIMIT 20;
    """
    df_counties = pd.read_sql_query(query, conn)
    conn.close()

    
    # Plot the data
    plt.figure(figsize=(12, 6))
    sns.barplot(x="Total_Deaths", y="County", hue="County", data=df_counties, palette="Reds_r", legend=False)

    plt.title("Top 20 US Counties by COVID-19 Deaths")
    plt.xlabel("Total Deaths")
    plt.ylabel("County")
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.show()

    
# ---- ESTIMATION OF PARAMETERS ----
def estimation_of_parameters(df, country):
    """
    Estimates key epidemiological parameters from the dataset:
    - Beta (transmission rate)
    - Gamma (recovery rate, assumed as 1/4.5)
    - Mu (mortality rate)
    - R0 (Basic Reproduction Number)
    - Alpha (additional recovery factor)
    
    Parameters:
    df : DataFrame containing COVID-19 case information.
    country : str : Name of the country for which the parameters are estimated.

    Returns:
    None (prints estimated values and plots R0 over time).
    """
    
    # Ensure the dataframe is sorted by date
    df = df.sort_values("Date").reset_index(drop=True)
    
    # Ensure necessary columns exist
    required_columns = {"Date", "Active", "Recovered", "Deaths", "Confirmed", "Population"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"Dataframe is missing required columns: {required_columns - set(df.columns)}")
    
    # Extract the total population from the first row (assuming constant population)
    N = df["Population"].iloc[0] if "Population" in df.columns else 1e6  # Default to 1M if missing

    # Fixed recovery rate
    gamma = 1 / 4.5  

    # Lists to store calculated parameters
    beta_estimates = []
    mu_estimates = []
    R0_estimates = []
    alpha_estimates = []

    for i in range(1, len(df)):
        new_cases = df["Confirmed"].iloc[i] - df["Confirmed"].iloc[i - 1]
        new_deaths = df["Deaths"].iloc[i] - df["Deaths"].iloc[i - 1]
        new_recovered = df["Recovered"].iloc[i] - df["Recovered"].iloc[i - 1]

        I_t = df["Active"].iloc[i]
        R_t = df["Recovered"].iloc[i]
        D_t = df["Deaths"].iloc[i]
        S_t = N - I_t - R_t - D_t  

        # Prevent division errors
        if I_t > 0 and S_t > 0:
            mu = new_deaths / I_t if I_t > 0 else 0
            mu_estimates.append(mu)

            beta = ((new_cases + mu * I_t + gamma * I_t) * N) / (S_t * I_t)
            beta_estimates.append(beta)

            alpha = (gamma * I_t - new_recovered) / R_t if R_t > 0 else 0
            alpha_estimates.append(alpha)

            R0 = beta / gamma
            R0_estimates.append(R0)

    # Compute average estimates
    avg_beta = np.mean(beta_estimates) if beta_estimates else 0
    avg_alpha = np.mean(alpha_estimates) if alpha_estimates else 0
    avg_mu = np.mean(mu_estimates) if mu_estimates else 0
    avg_R0 = np.mean(R0_estimates) if R0_estimates else 0

    # Print estimated values
    print(f"Estimated Parameters for {country}:")
    print(f"Estimated Beta: {avg_beta:.4f}")
    print(f"Estimated Alpha: {avg_alpha:.4f}")
    print(f"Estimated Mu: {avg_mu:.4f}")
    print(f"Estimated R0 (Basic Reproduction Number): {avg_R0:.4f}")
    
    # Plot the estimated R0 over time
    plt.plot(df["Date"].iloc[1:len(R0_estimates)+1], R0_estimates, label="Estimated R0", color="purple")
    plt.title(f"Estimated Basic Reproduction Number (R₀) in {country} Over Time")
    plt.xlabel("Date")
    plt.ylabel("R₀")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
def plot_cfr_over_time(df):
    """
    This function plots the Case Fatality Rate (CFR) over time.
    The Case Fatality Rate is calculated as CFR = Deaths / Confirmed cases.
    """
    # Calculate the Case Fatality Rate (CFR)
    df['CFR'] = df['Deaths'] / df['Confirmed']
    
    # Plot the CFR over time
    plt.figure(figsize=(10, 6))
    plt.plot(df["Date"], df["CFR"], label="Case Fatality Rate (CFR)", color='red')
    plt.xlabel("Date")
    plt.ylabel("CFR (%)")
    plt.title("Case Fatality Rate (CFR) Over Time")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_europe_active_cases(db_path="covid_database.db"):
    """
    Plots a choropleth map of Active COVID-19 Cases per Population in Europe.
    """

    # Connect to the database
    conn = sqlite3.connect(db_path)

    # Fetch relevant data
    query_europe = """
    SELECT "Country.Region" as country, ActiveCases, Population
    FROM worldometer_data
    WHERE Continent = 'Europe' AND Population > 0 AND ActiveCases IS NOT NULL;
    """
    df_europe = pd.read_sql_query(query_europe, conn)
    conn.close()

    # Calculate Active Cases per Population
    df_europe['active_cases_per_population'] = df_europe['ActiveCases'] / df_europe['Population']

    # Create the choropleth map
    fig = px.choropleth(
        df_europe,
        locations="country",
        locationmode="country names",
        color="active_cases_per_population",
        hover_name="country",
        color_continuous_scale=px.colors.sequential.Jet,
        title="🗺 Active COVID-19 Cases in Europe (per Population)",
        scope='europe'
    )
    fig.show()
 
# ---- RUNNING THE CODE ----

# Load the CSV into SQLite and process data
csv_path = "complete.csv"  # Update this path if needed
fix_missing_values(csv_path)

# Fetch country-specific data
selected_country = 'Bulgaria'
selected_region = 'Europe'
df_country = fetch_country_data(selected_country)
df_region = fetch_region_data(selected_region)
df_global = fetch_global_data()

population = fetch_population(selected_country)

# Generate graphs

# ---- COVID TRENDS ---- #

#covid_trends_country(df_country, selected_country)
covid_trends_region(df_region, selected_region)
#covid_trends_global(df_global)

# ---- ACTIVE VS RECOVERED VS DEATHS ---- #
#active_recovered_deaths_country(df_country, selected_country)
active_recovered_deaths_region(df_region, selected_region)
#active_recovered_deaths_global(df_global)

# ---- DEATH PLOTS COUNTY AND CONTINENT ---- #
#plot_death_rate_by_continent()
#plot_deaths_by_us_county()
#plt.show()

# ---- ESTIMATION OF PARAMETERS ---- #
if population:
    df_country["Population"] = population
    estimation_of_parameters(df_country, selected_country)
else:
    print(f"Population data for {selected_country} not found.")
    
# ---- CASE FATALITY RATE ---- #
plot_cfr_over_time(df_country)

plot_europe_active_cases()
