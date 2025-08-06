import numpy as np
import pandas as pd
import os
import logging
# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.callbacks import EarlyStopping
from graphing import *
from sklearn.metrics import mean_absolute_error
from sklearn.utils import shuffle


def create_sequences(X, y, seq_len: int =90):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len):
        X_seq.append(X[i:i + seq_len])  # last N days
        y_seq.append(y[i + seq_len])  # next day label
    return np.array(X_seq), np.array(y_seq)

def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))

def divide_data(df_scaled: pd.DataFrame, SEQ_LEN: int, train_part: float = 0.8):
    # Use your already scaled df_scaled (features only)
    feature_cols = [c for c in df_scaled.columns if c != "Result"]
    X_raw = df_scaled[feature_cols].values
    y_raw = df_scaled["Result"].values

    X_seq, y_seq = create_sequences(X_raw, y_raw, seq_len=SEQ_LEN)

    X_seq, y_seq = shuffle(X_seq, y_seq, random_state=9073)

    # Train/test split (keep time order)
    split_idx = int(len(X_seq) * train_part)
    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
    return X_train, X_test, y_train, y_test

def fast_model_evaluation(history, ticker, model, X_test, y_test):
    plot_training_history(history, ticker)

    y_pred = model.predict(X_test).ravel()
    mae = mean_absolute_error(y_test, y_pred)
    print(f"\nMean Absolute Error: {mae:.5f}")

    for i in range(5):
        print(f"Actual: {y_test[i]:.4f}  |  Predicted: {y_pred[i]:.4f}")

    under = sum(y_pred < y_test)
    over = sum(y_pred > y_test)
    print(f"Underpredictions: {under}  |  Overpredictions: {over}")

def train_model(X_train, X_test, y_train, y_test, SEQ_LEN: int, ticker):
    # GRU Model for regression
    model = models.Sequential([
        Input(shape=(SEQ_LEN, X_train.shape[2])),
        layers.GRU(256, return_sequences=True, dropout=0.2, recurrent_dropout=0.1),
        layers.LayerNormalization(),
        layers.GRU(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.1),
        layers.LayerNormalization(),
        layers.GRU(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.1),
        layers.LayerNormalization(),
        layers.GRU(32, dropout=0.1, recurrent_dropout=0.05),
        layers.BatchNormalization(),
        layers.Dense(16, activation=mish),
        layers.Dropout(0.3),
        layers.Dense(8, activation=mish),
        layers.Dropout(0.2),
        layers.Dense(4, activation=mish),
        layers.Dense(1)  # Linear output for regression
    ])

    @tf.keras.utils.register_keras_serializable()
    class ConservativeLoss(tf.keras.losses.Loss):
        def __init__(self, alpha=7.5, name="conservative_loss"):
            super().__init__(name=name)
            self.alpha = alpha

        def call(self, y_true, y_pred):
            error = y_pred - y_true
            over_penalty = tf.square(tf.maximum(error, 0.0)) * self.alpha
            under_penalty = tf.square(tf.minimum(error, 0.0))
            return tf.reduce_mean(over_penalty + under_penalty)

        def get_config(self):
            return {"alpha": self.alpha}

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss=ConservativeLoss(alpha=5),
        metrics=['mae']
    )

    model.summary()

    # Early stopping to prevent overfitting
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=25,
        restore_best_weights=True
    )

    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=120,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )

    dir_path = f"./Infos/{ticker}"
    os.makedirs(dir_path, exist_ok=True)
    model_path = os.path.join(dir_path, f"model_{ticker}.keras")
    model.save(model_path)

    fast_model_evaluation(history, ticker, model, X_test, y_test)

    return model, history
