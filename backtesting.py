import json
from graphing import *
from preProcess import *
from typing import List
import yfinance as yf
import math

def get_prepare_data(ticker:str, intervals: List[str], SEQ_LEN: int, scaler: StandardScaler):
    df_back = yf.download(ticker, start="2010-01-01", end="2024-12-31", interval="1d")
    if isinstance(df_back.columns, pd.MultiIndex):
        df_back.columns = [col[0] for col in df_back.columns]
    df_back = df_back.reset_index()

    df_back = add_past_updown_columns(df_back, intervals)
    df_back = add_continuous_columns(df_back, intervals)

    df_back = add_multi_RSI(df_back)
    df_back = add_multi_heikin_ashi_rsi(df_back)
    df_back = add_multi_macd(df_back)
    df_back = add_multi_bollinger(df_back)
    df_back = df_back.dropna()
    df_back = df_back.drop(["HA_Close"], axis=1)

    cols_to_shift = ['Volume'] + [c for c in df_back.columns if
                                  c.startswith("Continues") or c.startswith("RSI") or c.startswith(
                                      "Dist") or c.startswith("MACD") or c.startswith("BB")]
    df_back = shift_columns(df_back, cols_to_shift)

    df_back = scale_data(df_back, scaler)

    feature_cols = [c for c in df_back.columns if c not in ["Date", "Close", "High", "Low", "Open"]]
    X_backtest_seq = create_sequences_for_prediction(df_back[feature_cols].values, SEQ_LEN=SEQ_LEN)
    return df_back, X_backtest_seq

def execute_backtesting(df_back):
    results = []
    # RGS: Realistic Gain Score
    best_info = {'performance': float('-inf'), 'balance':[], 'quantile': 0, 'weighted_return': float('-inf'), "rgs": float("-inf")}

    for q in range(0, 100):
        initial_cash = 1000
        balance = [initial_cash]
        hits = 0
        trades = 0
        no_trade_days = 0
        returns_sum = 0
        hit_return_sum = 0

        threshold = df_back['PredictedHighChange'].quantile(q / 100)
        fee = 0
        position_size = 1  # full capital per trade

        for idx in range(len(df_back) - 1):
            row = df_back.iloc[idx]
            next_row = df_back.iloc[idx + 1]

            predicted = row["PredictedHigh"]
            predicted_change = row['PredictedHighChange']
            actual = row["High"]
            open_price = row["Open"]
            next_open_price = next_row["Open"]
            movement_fees = 0

            # Only trade if the prediction is good enough
            if predicted_change < threshold:
                balance.append(balance[-1])
                no_trade_days += 1
                continue
                # Simulate buy at open price

            trades += 1
            stock_amount = (position_size * balance[-1]) / open_price
            buy_outcome = stock_amount * open_price
            movement_fees += fee

            # If actual high reaches or exceeds predicted, it was a hit
            hit = actual >= predicted
            hits += int(hit)

            # Simulate sell — conservative: assume we sold at target or open+0.1% only if hit
            if hit:
                sell_income = predicted * stock_amount
                movement_fees += fee
            else:
                # conservative fallback: flat return, small loss from fees
                sell_income = next_open_price * stock_amount
                movement_fees += fee

            pnl = sell_income - buy_outcome - movement_fees
            returns_sum += (100 * pnl) / balance[-1]

            if hit:
                hit_return_sum += (100 * pnl) / balance[-1]

            balance.append(balance[-1] + pnl)

        profit_factor = get_profit_factor(balance)
        final_return = (((balance[-1] * 100) / initial_cash) - 100)
        weighted_return = final_return * max((profit_factor-1), 0)
        hit_rate = hits/trades
        average_pnl_per_trade = returns_sum / trades
        average_pnl_per_hit = hit_return_sum / hits
        rgs = returns_sum * math.log1p(max(profit_factor - 1, 0)) * math.log1p(hit_rate)
        results.append(
            {'return': final_return,
             'trades': trades,
             'no_trades': no_trade_days,
             'hit_rate': hit_rate,
             'profit_factor': profit_factor,
             'weighted_return': weighted_return,
             'average_pnl%_per_trade': average_pnl_per_trade,
             'average_pnl%_per_hit': average_pnl_per_hit,
             'rgs': rgs})
        performance = final_return / 100
        if weighted_return > best_info['weighted_return']:
            best_info['performance'] = performance
            best_info['weighted_return'] = weighted_return
            best_info['quantile'] = q / 100
            best_info['average_pnl%_per_trade']  = average_pnl_per_trade
            best_info['trades'] = trades
            best_info['no_trades'] = no_trade_days
            best_info['hit'] = hits
            best_info['hit_rate'] = hit_rate
            best_info['threshold'] = df_back['PredictedHighChange'].quantile(best_info['quantile'])
            best_info['balance'] = balance
            best_info['rgs'] = rgs
            best_info['average_pnl%_per_hit'] = average_pnl_per_hit

    best_threshold = df_back['PredictedHighChange'].quantile(best_info['quantile'])
    print(f"Best performance: {best_info['performance']:.2%}")
    print(f"Best quantile: {best_info['quantile']:.2f}")
    print(f"Best threshold: {best_threshold}")

    return best_info, results

def simple_backtesting(df_back, quantile):
    results = []
    best_info = {'performance': float('inf'), 'balance': [], 'quantile': 0, 'weighted_return': float('-inf'), 'rgs': float("-inf")}

    initial_cash = 1000
    balance = [initial_cash]
    hits = 0
    trades = 0
    no_trade_days = 0
    returns_sum = 0
    hit_return_sum = 0

    threshold = df_back['PredictedHighChange'].quantile(quantile)
    fee = 0
    position_size = 1  # full capital per trade

    for idx in range(len(df_back) - 1):
        row = df_back.iloc[idx]
        next_row = df_back.iloc[idx + 1]

        predicted = row["PredictedHigh"]
        predicted_change = row['PredictedHighChange']
        actual = row["High"]
        open_price = row["Open"]
        next_open_price = next_row["Open"]
        movement_fees = 0

        # Only trade if the prediction is good enough
        if predicted_change < threshold:
            balance.append(balance[-1])
            no_trade_days += 1
            continue
            # Simulate buy at open price

        trades += 1
        stock_amount = (position_size * balance[-1]) / open_price
        buy_outcome = stock_amount * open_price
        movement_fees += fee

        # If actual high reaches or exceeds predicted, it was a hit
        hit = actual >= predicted
        hits += int(hit)

        # Simulate sell — conservative: assume we sold at target or open+0.1% only if hit
        if hit:
            sell_income = predicted * stock_amount
            movement_fees += fee
        else:
            # conservative fallback: flat return, small loss from fees
            sell_income = next_open_price * stock_amount
            movement_fees += fee

        pnl = sell_income - buy_outcome - movement_fees
        returns_sum += (100 * pnl) / balance[-1]

        if hit:
            hit_return_sum += (100 * pnl) / balance[-1]

        balance.append(balance[-1] + pnl)

    profit_factor = get_profit_factor(balance)
    final_return = (((balance[-1] * 100) / initial_cash) - 100)
    weighted_return = final_return * max((profit_factor-1), 0)
    hit_rate = hits / trades
    average_pnl_per_trade = returns_sum / trades
    average_pnl_per_hit = hit_return_sum / hits
    rgs = returns_sum * math.log1p(max(profit_factor - 1, 0)) * math.log1p(hit_rate)
    results.append(
        {'return': final_return,
         'trades': trades,
         'no_trades': no_trade_days,
         'hit_rate': hit_rate,
         'profit_factor': profit_factor,
         'weighted_return': weighted_return,
         'average_pnl%_per_trade': average_pnl_per_trade,
         'average_pnl%_per_hit': average_pnl_per_hit,
         'rgs': rgs})
    performance = final_return / 100

    best_info['average_pnl%_per_trade'] = average_pnl_per_trade
    best_info['performance'] = performance
    best_info['weighted_return'] = weighted_return
    best_info['quantile'] = quantile
    best_info['trades'] = trades
    best_info['no_trades'] = no_trade_days
    best_info['hit'] = hits
    best_info['hit_rate'] = hit_rate
    best_info['threshold'] = df_back['PredictedHighChange'].quantile(best_info['quantile'])
    best_info['balance'] = balance
    best_info['rgs'] = rgs
    best_info['average_pnl%_per_hit'] = average_pnl_per_hit

    best_threshold = df_back['PredictedHighChange'].quantile(best_info['quantile'])
    print(f"Best performance: {best_info['performance']:.2%}")
    print(f"Best quantile: {best_info['quantile']:.2f}")
    print(f"Best threshold: {best_threshold}")

    return best_info, results

def get_max_drawdown(balance):
    peak = balance[0]
    max_dd = 0
    for val in balance:
        if val > peak:
            peak = val
        drawdown = (peak - val) / peak
        max_dd = max(max_dd, drawdown)

    return max_dd

def get_profit_factor(balance):
    losses = []
    gains = []
    last_val = balance[0]

    for val in balance[1:]:
        if (val == last_val):
            last_val = val
            continue

        change = val - last_val
        if (val < last_val):
            losses.append(change)

        if (val > last_val):
            gains.append(change)

        last_val = val

    profit_factor = sum(gains) / abs(sum(losses))
    return profit_factor

def get_average_gain(balance, number_trades):
    losses = []
    gains = []
    last_val = balance[0]

    for val in balance[1:]:
        if (val == last_val):
            last_val = val
            continue

        change = val - last_val
        if (val < last_val):
            losses.append(change)

        if (val > last_val):
            gains.append(change)

        last_val = val

    avg_gain_per_trade = (balance[-1] - balance[0]) / number_trades
    return avg_gain_per_trade

def complete_backtest(ticker, intervals, SEQ_LEN, scaler, model, quantile=None):
    df_back, X_backtest_seq = get_prepare_data(ticker, intervals=intervals, SEQ_LEN=SEQ_LEN, scaler=scaler)
    preds = model.predict(X_backtest_seq)

    df_back = df_back.iloc[SEQ_LEN - 1:].copy()
    df_back["PredictedHigh"] = preds.ravel()

    df_back.dropna(inplace=True)
    df_back["HighChange"] = ((100 * df_back["High"]) / df_back["Open"]) - 100
    df_back["PredictedHighChange"] = df_back["PredictedHigh"] * 100
    df_back["PredictedHigh"] = ((df_back['PredictedHighChange'] + 100) * df_back['Open']) / 100

    if quantile is not None:
        back_info, threshold_results = simple_backtesting(df_back, quantile)
    else:
        back_info, threshold_results = execute_backtesting(df_back)
        plot_metrics(threshold_results, ticker)

    back_info['max_drawdown'] = get_max_drawdown(back_info['balance'])
    back_info['profit_factor'] = get_profit_factor(back_info['balance'])

    plot_balance(df_back, back_info['balance'], ticker)

    dir_path = f"./Infos/{ticker}"
    os.makedirs(dir_path, exist_ok=True)
    info_path = os.path.join(dir_path, f"info_{ticker}.json")
    with open(info_path, "w") as f:
        json.dump(back_info, f, indent=4)
