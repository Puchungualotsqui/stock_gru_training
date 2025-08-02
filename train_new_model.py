import sys
from modelFunctions import *
from backtesting import *
import os

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python download_stock.py TICKER")
        sys.exit(1)
    ticker = sys.argv[1].upper()

    dir_path = f"./Infos/{ticker}"
    os.makedirs(dir_path, exist_ok=True)

    intervals = ["1d", "2d", "3d", "4d", "5d", "2w", "1M", "1y"]

    # Loading the data from Yahoo Finance
    df = get_yahoo_info(ticker)
    df = process_raw_data(df, intervals)
    df_scaled, scaler = scale_volume_and_continues(df, ticker)

    # Train the GRU model
    SEQ_LEN = 90

    X_train, X_test, y_train, y_test = divide_data(df_scaled, SEQ_LEN=SEQ_LEN)
    model, history = train_model(X_train, X_test, y_train, y_test, SEQ_LEN=SEQ_LEN, ticker=ticker)

    # Backtesting
    complete_backtest(ticker, intervals, SEQ_LEN, scaler, model)