import requests
import datetime as dt
import polars as pl
import sf_quant.data as sfd


def get_fund_weights(date_: dt.date) -> pl.DataFrame:
    url = "https://prod-api.silverfund.byu.edu"
    endpoint = "/all-holdings/summary"
    json = {"start": str(date_), "end": str(date_), "fund": "quant_paper"}

    response = requests.post(url + endpoint, json=json)

    if not response.ok:
        raise Exception(response.text)

    holdings = pl.DataFrame(response.json()["holdings"])

    return (
        holdings.filter(pl.col("active"))
        .select(
            "ticker", pl.col("value").truediv(pl.col("value").sum()).alias("weight")
        )
        .sort("ticker")
    )


def get_bai_tickers() -> list[str]:
    return pl.read_csv("data/bai_weights.csv")["ticker"].to_list()


def get_covariance_matrix(date_: dt.date, barrids: list[str]) -> pl.DataFrame:
    return sfd.construct_covariance_matrix(date_, barrids)


def get_ticker_barrid_mapping(date_: dt.date, tickers: list[str]) -> pl.DataFrame:
    return sfd.load_assets_by_date(
        date_=date_, in_universe=False, columns=["barrid", "ticker"]
    ).filter(pl.col("ticker").is_in(tickers))


def get_bai_weights(tickers: list[str]) -> pl.DataFrame:
    return (
        pl.read_csv("data/bai_weights.csv")
        .filter(pl.col("ticker").is_in(tickers))
        .with_columns(pl.col("weight").truediv(pl.col("weight").sum()))
    )


def get_bai_betas(
    weights: pl.DataFrame, covariance_matrix: pl.DataFrame
) -> pl.DataFrame:
    barrids = weights["barrid"].to_list()
    weights_np = weights["weight"].to_numpy()
    covariance_matrix_np = covariance_matrix.drop("barrid").to_numpy()
    betas = (
        weights_np.T
        @ covariance_matrix_np
        / (weights_np.T @ covariance_matrix_np @ weights_np)
    )
    return pl.DataFrame({"barrid": barrids, "beta": betas})


def get_fund_ai_beta(weights: pl.DataFrame, betas: pl.DataFrame) -> float:
    return (
        weights.join(betas, on="ticker", how="left")
        .select(pl.col("beta").mul(pl.col("weight")).sum())["beta"]
        .item()
    )


if __name__ == "__main__":
    date_ = dt.date(2025, 10, 30)
    tickers = get_bai_tickers()

    ticker_barrid_mapping = get_ticker_barrid_mapping(date_, tickers)
    tickers = ticker_barrid_mapping["ticker"].to_list()
    barrids = ticker_barrid_mapping["barrid"].to_list()

    covariance_matrix = get_covariance_matrix(date_, barrids)

    bai_weights = (
        get_bai_weights(tickers)
        .join(ticker_barrid_mapping, on="ticker", how="left")
        .select("barrid", "weight")
        .sort("barrid")
    )

    bai_betas = (
        get_bai_betas(weights=bai_weights, covariance_matrix=covariance_matrix)
        .join(ticker_barrid_mapping, on="barrid", how="left")
        .select("ticker", "beta")
        .sort("ticker")
    )

    fund_weights = get_fund_weights(date_)
    fund_ai_beta = get_fund_ai_beta(fund_weights, bai_betas)

    print(f"Ex ante AI beta: {fund_ai_beta:.2f}")
