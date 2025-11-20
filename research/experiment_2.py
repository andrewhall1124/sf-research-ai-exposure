import requests
import datetime as dt
import polars as pl
from rich import print
import yfinance as yf
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def get_fund_returns(start: dt.date, end: dt.date) -> pl.DataFrame:
    url = "https://prod-api.silverfund.byu.edu"
    endpoint = "/portfolio/time-series"
    json = {"start": str(start), "end": str(end), "fund": "quant_paper"}

    response = requests.post(url + endpoint, json=json)

    if not response.ok:
        raise Exception(response.text)

    return (
        pl.DataFrame(response.json()["records"])
        .with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"))
        .select("date", pl.col("return_").truediv(100).alias("fund_return"))
    )


def get_bai_returns(start: dt.date, end: dt.date) -> pl.DataFrame:
    raw = pl.DataFrame(
        yf.download(tickers=["BAI"], start=start, end=end, auto_adjust=True)
        .stack(future_stack=True)
        .reset_index()
    )

    return (
        raw.rename({col: col.lower() for col in raw.columns})
        .with_columns(pl.col("date").dt.date())
        .sort("date")
        .with_columns(pl.col("close").pct_change().alias("ai_return"))
        .drop_nulls()
        .sort("date")
        .select("date", "ai_return")
    )


def get_ai_beta(
    fund_returns: pl.DataFrame, ai_returns: pl.DataFrame, verbose: bool = False
) -> float:
    merged = fund_returns.join(ai_returns, on="date", how="left")
    model = smf.ols(formula="fund_return ~ ai_return", data=merged)
    results = model.fit()

    if verbose:
        print(results.summary())

    return results.params["ai_return"]


def create_regression_chart(
    fund_returns: pl.DataFrame, ai_returns: pl.DataFrame
) -> None:
    merged = (
        fund_returns.join(ai_returns, on="date", how="left")
        .with_columns(pl.col("fund_return", "ai_return").mul(100))
        .drop_nulls()
    )

    plt.figure(figsize=(10, 6))

    sns.scatterplot(merged.to_pandas(), x="ai_return", y="fund_return")

    # Add line of best fit
    x = merged["ai_return"].to_numpy()
    y = merged["fund_return"].to_numpy()
    coeffs = np.polyfit(x, y, 1)
    line_x = np.array([merged["ai_return"].min(), merged["ai_return"].max()])
    line_y = coeffs[0] * line_x + coeffs[1]
    plt.plot(line_x, line_y, "r-", linewidth=2)

    plt.xlabel("AI Daily Return (%)")
    plt.ylabel("Fund Daily Return (%)")

    plt.xlim([-5, 5])
    plt.ylim([-5, 5])

    plt.show()

def create_returns_chart(fund_returns: pl.DataFrame, ai_returns: pl.DataFrame) -> None:
    merged = (
        fund_returns.join(ai_returns, on="date", how="left")
        .with_columns(pl.col("fund_return", "ai_return"))
        .drop_nulls()
        .sort('date')
        .with_columns(
            pl.col('fund_return', 'ai_return').log1p().cum_sum().exp().sub(1).mul(100)
        )
    )

    plt.figure(figsize=(10, 6))

    sns.lineplot(merged, x='date', y='fund_return', label='Fund')
    sns.lineplot(merged, x='date', y='ai_return', label='AI')

    plt.xlabel(None)
    plt.ylabel("Cumulative Product Returns (%)")

    plt.show()

if __name__ == "__main__":
    start = dt.date(2025, 10, 23)
    end = dt.date(2025, 11, 19)

    fund_returns = get_fund_returns(start, end)
    bai_returns = get_bai_returns(start, end)
    beta = get_ai_beta(fund_returns, bai_returns)

    create_regression_chart(fund_returns, bai_returns)
    create_returns_chart(fund_returns, bai_returns)

    print(f"Ex post AI beta: {beta:.2f}")
