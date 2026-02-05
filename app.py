from data_history import load_price_history
from indicators import add_indicators
from plot_chart import plot_chart

if __name__ == "__main__":
    ticker = "BTC-USD"
    df = load_price_history(ticker)
    df = add_indicators(df, ticker)
    plot_chart(df)