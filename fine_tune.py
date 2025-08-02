import sys
from modelFunctions import *
import tensorflow as tf
from tensorflow.keras.models import load_model
from backtesting import *


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python download_stock.py MODEL_PATH SCALER_PATH TICKER")
        sys.exit(1)
    ticker = sys.argv[1].upper()
    model_path = sys.argv[2]
    scaler_path = sys.argv[3]

    dir_path = f"./Infos/{ticker}"
    os.makedirs(dir_path, exist_ok=True)

    intervals = ["1d", "2d", "3d", "4d", "5d", "2w", "1M", "1y"]

    df = get_yahoo_info(ticker)
    df = process_raw_data(df, intervals)

    scaler = joblib.load(scaler_path)
    df_scaled = scale_data(df, scaler)

    SEQ_LEN = 90
    X_train, X_test, y_train, y_test = divide_data(df_scaled, SEQ_LEN=SEQ_LEN)

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


    model_base = load_model(model_path, custom_objects={"ConservativeLoss": ConservativeLoss})

    # Ensure all layers are trainable
    for layer in model_base.layers:
        layer.trainable = True

    # Recompile with lower learning rate
    model_base.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=ConservativeLoss(alpha=7.5),
        metrics=['mae']
    )

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=25,
        restore_best_weights=True
    )

    # Fine-tune training
    fine_tune_history = model_base.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )

    fast_model_evaluation(fine_tune_history, ticker, model_base, X_test, y_test)

    model_path = os.path.join(dir_path, f"model_{ticker}.keras")
    model_base.save(model_path)

    # Backtesting
    complete_backtest(ticker, intervals, SEQ_LEN, scaler, model_base)