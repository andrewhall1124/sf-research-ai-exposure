import requests
import datetime as dt
import polars as pl
from rich import print
import yfinance as yf
import statsmodels.formula.api as smf


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


def get_ai_beta(fund_returns: pl.DataFrame, ai_returns: pl.DataFrame, verbose: bool = False) -> float:
    merged = fund_returns.join(ai_returns, on="date", how="left")
    model = smf.ols(formula="fund_return ~ ai_return", data=merged)
    results = model.fit()

    if verbose:
        print(results.summary())

    return results.params['ai_return']


if __name__ == "__main__":
    start = dt.date(2025, 10, 23)
    end = dt.date(2025, 11, 19)

    fund_returns = get_fund_returns(start, end)
    bai_returns = get_bai_returns(start, end)
    beta = get_ai_beta(fund_returns, bai_returns)

    print(f"Ex post AI beta: {beta:.2f}")
