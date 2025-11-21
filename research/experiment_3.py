import requests
import datetime as dt
import polars as pl
import sf_quant.data as sfd

DATE = dt.date(2025, 10, 30)


def get_fund_weights() -> pl.DataFrame:
    url = "https://prod-api.silverfund.byu.edu"
    endpoint = "/all-holdings/summary"
    json = {"start": str(DATE), "end": str(DATE), "fund": "quant_paper"}

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


def get_covariance_matrix(tickers: list[str]) -> pl.DataFrame:
    ticker_barrid_mapping = get_ticker_barrid_mapping()

    ticker_list = ticker_barrid_mapping["ticker"].to_list()
    barrid_list = ticker_barrid_mapping["barrid"].to_list()

    barrid_to_ticker = {"barrid": "ticker"} | {
        barrid: ticker for barrid, ticker in zip(barrid_list, ticker_list)
    }

    barrids = (
        ticker_barrid_mapping.filter(pl.col("ticker").is_in(tickers))["barrid"]
        .sort()
        .to_list()
    )

    covariance_matrix = (
        sfd.construct_covariance_matrix(DATE, barrids)
        .rename(barrid_to_ticker, strict=False)
        .with_columns(pl.col("ticker").replace(barrid_to_ticker))
        .select("ticker", *sorted(tickers))
        .sort("ticker")
    )

    return covariance_matrix


def get_ticker_barrid_mapping() -> pl.DataFrame:
    return sfd.load_assets_by_date(
        date_=DATE, in_universe=True, columns=["barrid", "ticker"]
    )


def get_bai_weights(tickers: list[str]) -> pl.DataFrame:
    return (
        pl.read_csv("data/bai_weights.csv")
        .filter(pl.col("ticker").is_in(tickers))
        .with_columns(pl.col("weight").truediv(pl.col("weight").sum()))
        .sort("ticker")
    )


def get_bai_betas(
    weights: pl.DataFrame, covariance_matrix: pl.DataFrame
) -> pl.DataFrame:
    tickers = weights["ticker"].to_list()
    weights_np = weights["weight"].to_numpy()
    covariance_matrix_np = covariance_matrix.drop("ticker").to_numpy()
    betas = (
        weights_np.T
        @ covariance_matrix_np
        / (weights_np.T @ covariance_matrix_np @ weights_np)
    )
    return pl.DataFrame({"ticker": tickers, "beta": betas})


def get_active_ai_beta(weights: pl.DataFrame, betas: pl.DataFrame) -> float:
    merged = betas.join(weights, on="ticker", how="left")

    print(merged.drop_nulls().sort('weight', descending=True))

    return merged.select(pl.col("beta").mul(pl.col("weight")).sum())["beta"].item()


def get_benchmarket_weights() -> pl.DataFrame:
    ticker_barrid_mapping = get_ticker_barrid_mapping()
    return (
        sfd.load_benchmark(DATE, DATE)
        .join(ticker_barrid_mapping, on="barrid", how="left")
        .select("ticker", pl.col("weight").alias("benchmark_weight"))
        .sort("ticker")
    )


def get_active_weights(
    fund_weights: pl.DataFrame, benchmark_weights: pl.DataFrame
) -> pl.DataFrame:
    return (
        fund_weights
        .join(benchmark_weights, on='ticker', how='left').with_columns(
            pl.col('weight').sub('benchmark_weight').alias('active_weight')
        )
        # .select('ticker', pl.col('active_weight').truediv(pl.col('active_weight').sum()).alias('weight'))
        .select('ticker', pl.col('active_weight').alias('weight'))
    )


def get_valid_tickers(tickers: list[str]) -> list[str]:
    ticker_barrid_mapping = get_ticker_barrid_mapping()
    return ticker_barrid_mapping.filter(pl.col("ticker").is_in(tickers))[
        "ticker"
    ].to_list()


if __name__ == "__main__":
    benchmark_weights = get_benchmarket_weights()

    tickers = get_bai_tickers()

    valid_tickers = get_valid_tickers(tickers)
    covariance_matrix = get_covariance_matrix(tickers=valid_tickers)

    bai_weights = get_bai_weights(valid_tickers)
    bai_betas = get_bai_betas(weights=bai_weights, covariance_matrix=covariance_matrix)

    fund_weights = get_fund_weights()
    active_weights = get_active_weights(fund_weights, benchmark_weights)
    fund_ai_beta = get_active_ai_beta(active_weights, bai_betas)

    print(f"Ex ante active AI beta: {fund_ai_beta:.2f}")
