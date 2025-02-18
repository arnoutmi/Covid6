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

#covid_trends(df)


def covid_trends_filtered(df, start_date, end_date):
    df["Date"] = pd.to_datetime(df["Date"])
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    df_filtered = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)]

    covid_trends(df_filtered)

#covid_trends_filtered(df, "2020-03-01", "2020-06-01")



