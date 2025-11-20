import requests
import datetime as dt
import polars as pl

date_ = str(dt.date(2025, 11, 18))

url = "https://prod-api.silverfund.byu.edu"
endpoint = "/all-holdings/summary"
json = {"start": date_, "end": date_, "fund": "quant_paper"}

response = requests.post(url + endpoint, json=json)

if not response.ok:
    raise Exception(response.text)

holdings = pl.DataFrame(response.json()["holdings"])

print(holdings)
