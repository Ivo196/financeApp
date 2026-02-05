# data_history.py
import numpy as np
import pandas as pd
import yfinance as yf

def load_price_history(ticker: str, period: str = "10y", interval: str = "1wk") -> pd.DataFrame:

    df = yf.download(ticker,
                    period=period,
                    interval=interval,
                    auto_adjust=True, 
                    multi_level_index=False
                    )
    df = df.reset_index().rename(columns={"Date": "time", "Datetime": "time"})
    df = df.rename(columns={
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume",
    })

    df.to_csv(f"data/{ticker}.csv", index=False)

    print(f"Successfully downloaded {ticker} price history {df.tail()}")
    
    return df

