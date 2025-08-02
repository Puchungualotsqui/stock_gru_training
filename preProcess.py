from sklearn.preprocessing import StandardScaler
import numpy as np
import os
import joblib
from typing import List
from dataPulling import *

def scale_data(df, scaler):
    cols_to_scale = ['Volume'] + [c for c in df.columns if
                                  c.startswith("Continues") or c.startswith("RSI") or c.startswith(
                                      "Dist") or c.startswith("MACD") or c.startswith("BB")]
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])
    return df

def add_past_updown_columns(df: pd.DataFrame, intervals: list[str]) -> pd.DataFrame:
    """
    Much faster backward-looking UpDown calculation using lists & pointers.
    """
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)

    dates = df['Date'].values
    opens = df['Open'].values
    closes = df['Close'].values
    n = len(df)

    for interval in intervals:
        col_name = f"UpDown{interval}"

        # Parse interval
        num = int(''.join(filter(str.isdigit, interval)))
        unit = ''.join(filter(str.isalpha, interval)).lower()
        if unit == 'd':
            offset = pd.DateOffset(days=num)
        elif unit == 'w':
            offset = pd.DateOffset(weeks=num)
        elif unit == 'm':
            offset = pd.DateOffset(months=num)
        elif unit == 'y':
            offset = pd.DateOffset(years=num)
        else:
            raise ValueError(f"Unsupported interval: {interval}")

        # Precompute date offsets (array of required past dates)
        required_past_dates = pd.to_datetime(df['Date']) - offset
        required_past_dates = required_past_dates.values

        result = [None] * n
        j = 0  # pointer for past rows

        for i in range(n):
            target_date = required_past_dates[i]

            # Move j forward until dates[j] <= target_date < dates[j+1]
            while j < n and dates[j] <= target_date:
                j += 1

            # The last valid past row is j-1
            if j > 0:
                past_idx = j - 1
                diff = closes[past_idx] - opens[past_idx]
                result[i] = 1 if diff > 0 else 0

        df[col_name] = result
    print(df.columns)
    # Drop rows that have NaNs for any interval
    df.dropna(subset=[f"UpDown{interval}" for interval in intervals], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def add_continuous_columns(df: pd.DataFrame, intervals: list[str]) -> pd.DataFrame:
    """
    For each UpDown<interval> column, adds a Continues<interval> column that counts
    how many days the trend has continued (+ for 1s, - for 0s).
    """
    df = df.copy()

    for interval in intervals:
        updown_col = f"UpDown{interval}"
        cont_col = f"Continues{interval}"

        if updown_col not in df.columns:
            raise ValueError(f"{updown_col} not found in dataframe")

        updown_values = df[updown_col].values
        n = len(updown_values)

        result = [0] * n
        count = 0

        for i in range(n):
            if i == 0:
                # First row, just assign +1 or -1
                count = 1 if updown_values[i] == 1 else -1
            else:
                if updown_values[i] == updown_values[i - 1]:
                    # Same trend, continue counting
                    count = count + 1 if count > 0 else count - 1
                else:
                    # Trend reversed, restart
                    count = 1 if updown_values[i] == 1 else -1
            result[i] = count

        df[cont_col] = result

    return df


def add_high_target_result(df_inside: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a 'Result' column:
    - The percentage increase from Open to High for the day.
    - Ensures non-negative values (clipped at 0).
    """
    df_inside = df_inside.copy()

    # Compute (High - Open) / Open
    result = (df_inside["High"] - df_inside["Open"]) / df_inside["Open"]

    # Replace negative values with 0
    df_inside["Result"] = result.clip(lower=0)

    return df_inside


def add_multi_RSI(df: pd.DataFrame, periods: list[int] = [7, 14, 28], price_col: str = "Close") -> pd.DataFrame:
    """
    Adds multiple RSI columns for different lookback periods based on `price_col`.

    RSI measures momentum:
      - >70 = overbought
      - <30 = oversold

    Parameters:
        df        : DataFrame with a price column (default "Close")
        periods   : List of RSI periods (e.g. [7, 14, 28])
        price_col : Column name with price

    Returns:
        DataFrame with new columns "RSI_<period>" for each period.
    """
    df = df.copy()

    for period in periods:
        delta = df[price_col].diff()

        # Separate gains and losses
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        # Use EMA for average gain/loss
        avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        df[f"RSI_{period}"] = rsi

    return df


def add_multi_heikin_ashi_rsi(
        df: pd.DataFrame,
        periods: list[int] = [7, 14, 28],
        buy_zone: float = 30.0,
        sell_zone: float = 70.0
) -> pd.DataFrame:
    """
    Adds Heikin Ashi Close and multi-timeframe RSI based on HA_Close.
    Also creates BuyZone/SellZone flags for each timeframe.

    BuyZone = 1 if RSI < buy_zone
    SellZone = 1 if RSI > sell_zone

    Parameters:
        df        : DataFrame with Open, High, Low, Close
        periods   : list of RSI lookback periods (default [7,14,28])
        buy_zone  : RSI threshold below which is considered 'buy'
        sell_zone : RSI threshold above which is considered 'sell'

    Returns:
        DataFrame with:
          - HA_Close
          - RSI_HA_<period>
          - RSI_HA_<period>_BuyZone
          - RSI_HA_<period>_SellZone
    """
    df = df.copy()

    # Heikin Ashi Close
    df['HA_Close'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4

    for period in periods:
        delta = df['HA_Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        # Save RSI
        rsi_col = f"RSI_HA_{period}"
        df[rsi_col] = rsi

        # Create BuyZone and SellZone flags
        df[f"{rsi_col}_BuyZone"] = (rsi < buy_zone).astype(int)
        df[f"{rsi_col}_SellZone"] = (rsi > sell_zone).astype(int)
    return df


def add_multi_macd(df: pd.DataFrame, macd_params=[(5, 20, 9), (12, 26, 9), (20, 50, 18)],
                   price_col="Close") -> pd.DataFrame:
    """
    Adds multiple MACD features for different EMA timeframes.

    macd_params = list of (fast, slow, signal)
    """
    df = df.copy()

    for fast, slow, signal in macd_params:
        # EMA calculations
        ema_fast = df[price_col].ewm(span=fast, adjust=False).mean()
        ema_slow = df[price_col].ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        macd_hist = macd - macd_signal

        # Add columns
        suffix = f"{fast}_{slow}_{signal}"
        df[f"MACD_{suffix}"] = macd
        df[f"MACDSignal_{suffix}"] = macd_signal
        df[f"MACDHist_{suffix}"] = macd_hist

    return df


def add_bollinger_bands(df: pd.DataFrame, period: int = 20, k: float = 2.0) -> pd.DataFrame:
    """
    Adds Bollinger Band features:
      - BB_MA_<period>: rolling mean
      - BB_Upper_<period>: upper band
      - BB_Lower_<period>: lower band
      - BB_Dist_<period>: normalized distance of Close to MA
      - BB_Width_<period>: band width normalized by MA
    """
    df = df.copy()

    ma = df['Close'].rolling(period).mean()
    std = df['Close'].rolling(period).std()

    upper = ma + k * std
    lower = ma - k * std

    df[f"BB_MA_{period}"] = ma
    df[f"BB_Upper_{period}"] = upper
    df[f"BB_Lower_{period}"] = lower

    # Normalized distance (0 = MA, +1 = upper band, -1 = lower band)
    df[f"BB_Dist_{period}"] = (df['Close'] - ma) / (k * std)

    # Band width as a % of MA (volatility measure)
    df[f"BB_Width_{period}"] = (upper - lower) / ma

    return df

def add_multi_bollinger(df: pd.DataFrame, periods=[10, 20, 50], k=2.0):
    for p in periods:
        df = add_bollinger_bands(df, period=p, k=k)
    return df


def shift_columns(df: pd.DataFrame, column_names: list[str]) -> pd.DataFrame:
    """
    Shifts the specified columns forward by 1 row to simulate real-world prediction.

    After shifting:
      - The first row becomes NaN (no past data)
      - We drop rows with NaN in any shifted column
    """
    df = df.copy()

    # Shift each column by 1
    for col in column_names:
        if col in df.columns:
            df[col] = df[col].shift(1)
        else:
            raise ValueError(f"Column '{col}' not found in DataFrame!")

    # Drop rows with NaN in any shifted column
    df.dropna(subset=column_names, inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def create_sequences_for_prediction(X, SEQ_LEN=90):
    X_seq = []
    for i in range(len(X) - SEQ_LEN + 1):
        X_seq.append(X[i:i + SEQ_LEN])
    return np.array(X_seq)

def process_raw_data(df_inside: pd.DataFrame, intervals: List[str]) -> pd.DataFrame:
    # Remove leading rows with Volume == 0
    df_inside = df_inside[df_inside['Volume'].ne(0)].reset_index(drop=True)

    # Find the first index where Open != Close
    first_diff_idx = df_inside.index[(df_inside['Open'] != df_inside['Close'])].min()
    # Keep only rows from that point onward
    if pd.notna(first_diff_idx):
        df_inside = df_inside.loc[first_diff_idx:].reset_index(drop=True)

    df_inside = add_past_updown_columns(df_inside, intervals)
    df_inside = add_continuous_columns(df_inside, intervals)

    df_inside = add_high_target_result(df_inside)
    df_inside = add_multi_RSI(df_inside)
    df_inside = add_multi_heikin_ashi_rsi(df_inside)
    df_inside = add_multi_macd(df_inside)
    df_inside = add_multi_bollinger(df_inside)

    df_inside.drop(["Date", "High", "Low", "Open", "Close", "HA_Close"], axis=1, inplace=True)

    cols_to_shift = ['Volume'] + [c for c in df_inside.columns if
                                  c.startswith("Continues") or c.startswith("RSI") or c.startswith(
                                      "Dist") or c.startswith("MACD") or c.startswith("BB")]
    df_inside = shift_columns(df_inside, cols_to_shift)

    return df_inside

def scale_volume_and_continues(df_internal: pd.DataFrame, ticker) -> (pd.DataFrame, StandardScaler):
    """
    Scales Volume and all Continues* columns using StandardScaler.
    Leaves UpDown* columns untouched.
    """
    df_internal_copied = df_internal.copy()

    # Find columns to scale
    cols_to_scale = ['Volume'] + [c for c in df_internal_copied.columns if
                                  c.startswith("Continues") or c.startswith("RSI") or c.startswith(
                                      "Dist") or c.startswith("MACD") or c.startswith("BB")]

    scaler = StandardScaler()
    df_internal_copied[cols_to_scale] = scaler.fit_transform(df_internal_copied[cols_to_scale])

    dir_path = f"./Infos/{ticker}"
    os.makedirs(dir_path, exist_ok=True)
    scaler_path = os.path.join(dir_path, f"scaler_{ticker}.pkl")
    joblib.dump(scaler, scaler_path)

    return df_internal_copied, scaler