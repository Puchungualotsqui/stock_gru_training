import yfinance as yf
import pandas as pd

def get_yahoo_info(ticker: str) -> pd.DataFrame:
    data = yf.download(ticker, period="max", interval="1d")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    df = data[["Open", "High", "Low", "Close", "Volume"]].reset_index()
    print(df.head())
    return df
