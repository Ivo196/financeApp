# indicators.py
import numpy as np
import pandas as pd
import pandas_ta as ta

def add_indicators(df: pd.DataFrame, ticker: str, price_col: str = "close") -> pd.DataFrame:

    df["time"] = pd.to_datetime(df["time"])
    close = df[price_col].astype(float)
    # EMA 20/50/200
    df.ta.ema(length=20, append=True)
    df.ta.ema(length=50, append=True)
    df.ta.ema(length=200, append=True)

    # MACD (12,26,9)
    df.ta.macd(append=True)

    # RSI 14
    df.ta.rsi(length=14, append=True)

    # Bollinger Bands (20,2)
    df.ta.bbands(length=20, std=2, append=True)

    #Add to csv
    df.to_csv(f"data/{ticker}_indicators.csv", index=False)
    print(f"Successfully added indicators to {ticker} price history {df.tail()}")

    return df