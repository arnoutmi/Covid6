import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

df = pd.read_csv("day_wise.csv")
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(by="Date")

def covid_trends(df):
    fig, axes = plt.subplots(3, 1, figsize=(15, 9))

    axes[0].plot(df["Date"], df["New cases"], color="blue", label="New Cases")
    axes[0].set_title("Daily New COVID-19 Cases")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Cases")
    axes[0].legend()

    axes[1].plot(df["Date"], df["New deaths"], color="red", label="New Deaths")
    axes[1].set_title("Daily COVID-19 Deaths")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Deaths")
    axes[1].legend()

    axes[2].plot(df["Date"], df["New recovered"], color="green", label="New Recoveries")
    axes[2].set_title("Daily Recoveries from COVID-19")
    axes[2].set_xlabel("Date")
    axes[2].set_ylabel("Recoveries")
    axes[2].legend()

    plt.tight_layout()
    plt.show()

#####covid_trends(df)


def covid_trends_filtered(df, start_date, end_date):
    df["Date"] = pd.to_datetime(df["Date"])
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    df_filtered = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)]

    covid_trends(df_filtered)

#####covid_trends_filtered(df, "2020-03-01", "2020-06-01")

def estimation_of_parameters(df):
    N = 17000000
    beta_estimates = []
    gamma_estimates = []
    mu_estimates = []
    R0_estimates = []
    for i in range(1, len(df)):
        new_cases = df["New cases"].iloc[i]
        new_deaths = df["New deaths"].iloc[i]
        new_recovered = df["New recovered"].iloc[i]
        
        I_t = df["Active"].iloc[i]  
        R_t = df["Recovered"].iloc[i] 
        D_t = df["Deaths"].iloc[i]
        S_t = N - I_t - R_t - D_t  
        
        if I_t > 0:
            beta = (new_cases * N) / (S_t * I_t)
            beta_estimates.append(beta)

            if I_t > 0:
                gamma = new_recovered / I_t
                gamma_estimates.append(gamma)

            if I_t > 0:
                mu = new_deaths / I_t
                mu_estimates.append(mu)

            if gamma > 0:
                R0 = beta / gamma
                R0_estimates.append(R0)

    average_beta = sum(beta_estimates) / len(beta_estimates) if beta_estimates else 0
    average_gamma = sum(gamma_estimates) / len(gamma_estimates) if gamma_estimates else 0
    average_mu = sum(mu_estimates) / len(mu_estimates) if mu_estimates else 0
    average_R0 = sum(R0_estimates) / len(R0_estimates) if R0_estimates else 0

    print(f"Estimated Beta: {average_beta}")
    print(f"Estimated Gamma: {average_gamma}")
    print(f"Estimated Mu: {average_mu}")
    print(f"Estimated R0 (Basic Reproduction Number): {average_R0}")

    plt.figure(figsize=(10, 6))
    plt.plot(df["Date"].iloc[1:], R0_estimates, label="Estimated R0", color="purple")
    plt.title("Estimated Basic Reproduction Number (R₀) Over Time")
    plt.xlabel("Date")
    plt.ylabel("R₀")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return average_beta, average_gamma, average_mu, average_R0


estimation_of_parameters(df)



