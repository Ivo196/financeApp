import pandas as pd
from lightweight_charts import Chart

def plot_chart(df: pd.DataFrame = None, ticker: str = "AAPL") -> None:
    
    df["time"] = df["time"].dt.strftime('%Y-%m-%d')

    chart = Chart(inner_width=1, inner_height=0.55)
    chart.time_scale(visible=False)  
    
    # Columns: time | open | high | low | close | volume 
    chart.set(df[["time","open","high","low","close","volume"]])

    # EMA 20
    ema20 = chart.create_line("EMA_20", color="yellow")
    ema20.set(df[["time", "EMA_20"]].dropna())

    # EMA 50
    ema50 = chart.create_line("EMA_50", color="blue")
    ema50.set(df[["time", "EMA_50"]].dropna())
    
    # EMA 200
    ema200 = chart.create_line("EMA_200", color="red")
    ema200.set(df[["time", "EMA_200"]].dropna())

    # --- MACD subpanel ---
    macd_chart = chart.create_subchart(width=1, height=0.25, sync=True)
    macd_chart.legend(True)

    # MACD
    macd = macd_chart.create_line("MACD_12_26_9", color="blue")
    macd.set(df[["time","MACD_12_26_9"]].dropna())

    # MACD Signal
    macd_signal = macd_chart.create_line("MACDs_12_26_9", color="red")
    macd_signal.set(df[["time","MACDs_12_26_9"]].dropna())

    # MACD Histogram
    macd_histogram = macd_chart.create_line("MACDh_12_26_9", color="green")
    macd_histogram.set(df[["time","MACDh_12_26_9"]].dropna())

    # --- referencia 0 para MACD como serie constante ---
    df["MACD 0"] = 0
    macd0 = macd_chart.create_line("MACD 0", color="white", width=1, style="dotted")
    macd0.set(df[["time", "MACD 0"]])

    # --- RSI subpanel ---
    rsi_chart = chart.create_subchart(width=1, height=0.20, sync=True)
    rsi_chart.legend(True)

    # RSI
    rsi = rsi_chart.create_line("RSI_14", color="pink")
    rsi.set(df[["time","RSI_14"]].dropna())

    # --- referencias 70/30 como series constantes ---
    df["RSI 70"] = 70
    df["RSI 30"] = 30

    rsi70 = rsi_chart.create_line("RSI 70", color="white", width=1, style="dotted")
    rsi70.set(df[["time", "RSI 70"]])

    rsi30 = rsi_chart.create_line("RSI 30", color="white", width=1, style="dotted")
    rsi30.set(df[["time", "RSI 30"]])

    
    
    chart.show(block=True)