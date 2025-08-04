import sys
import os
from dataPulling import *
import joblib
from backtesting import *
from preProcess import *
import tensorflow as tf
from tensorflow.keras.models import load_model


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python complete_backtest.py TICKER [MODEL_PATH] [SCALER_PATH] [THRESHOLD]")
        sys.exit(1)

    ticker = sys.argv[1].upper()

    # Handle optional arguments
    model_path = sys.argv[2] if len(sys.argv) > 2 else f"./Infos/{ticker}/model_{ticker}.keras"

    if len(sys.argv) > 3:
        scaler_path = sys.argv[3]
    else:
        scaler_path = f"./Infos/{ticker}/scaler_{ticker}.pkl"
        if not os.path.exists(scaler_path):
            fallback_path = "./Infos/scaler.pkl"
            if os.path.exists(fallback_path):
                scaler_path = fallback_path
            else:
                print(f"Error: Neither {scaler_path} nor fallback {fallback_path} exist.")
                sys.exit(1)

    quantile = float(sys.argv[4]) if len(sys.argv) > 4 else None

    dir_path = f"./Infos/{ticker}"
    os.makedirs(dir_path, exist_ok=True)

    intervals = ["1d", "2d", "3d", "4d", "5d", "2w", "1M", "1y"]

    df = get_yahoo_info(ticker)
    df = process_raw_data(df, intervals)

    scaler = joblib.load(scaler_path)
    df_scaled = scale_data(df, scaler)

    @tf.keras.utils.register_keras_serializable()
    class ConservativeLoss(tf.keras.losses.Loss):
        def __init__(self, alpha, name="conservative_loss"):
            super().__init__(name=name)
            self.alpha = alpha

        def call(self, y_true, y_pred):
            error = y_pred - y_true
            over_penalty = tf.square(tf.maximum(error, 0.0)) * self.alpha
            under_penalty = tf.square(tf.minimum(error, 0.0))
            return tf.reduce_mean(over_penalty + under_penalty)

        def get_config(self):
            return {"alpha": self.alpha}


    model = load_model(model_path, custom_objects={"ConservativeLoss": ConservativeLoss})

    SEQ_LEN = 90
    complete_backtest(ticker, intervals, SEQ_LEN, scaler, model, quantile=quantile)