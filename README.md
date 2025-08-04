# GRU Stock Predictor & Backtesting Pipeline

This project is a **complete GRU-based stock prediction and backtesting pipeline** using **Yahoo Finance data**.  
It includes data preprocessing, feature engineering, model training, fine-tuning, and threshold-based backtesting with performance visualization.

---

## What It Does

- Pulls historical OHLCV data using `yfinance`
- Computes custom features: RSI, MACD, Heikin-Ashi RSI, Bollinger Bands, UpDown trends, and continuations
- Prepares sequences for GRU time-series prediction
- Trains and fine-tunes a Keras GRU model with a custom **conservative loss function**
- Performs threshold-based backtesting
- Saves predictions, metrics, and plots to the `./Infos/{TICKER}` folder

---

## Directory Structure

```
├── dataPulling.py # Download OHLCV data from Yahoo Finance
├── preProcess.py # Feature engineering and scaling
├── modelFunctions.py # GRU architecture and training utils
├── train_new_model.py # Train from scratch
├── fine_tune.py # Fine-tune existing model
├── complete_backtest.py # Full backtesting script
├── simple_backtest.py # Threshold-only backtest
├── backtesting.py # Core backtesting logic
├── graphing.py # Training & result visualizations
├── Infos/ # Output directory for models & plots
```


---

## Requirements

Install required packages:

```
pip install yfinance joblib matplotlib scikit-learn tensorflow
```
## How to Use

### Train a New GRU Model

```bash
python train_new_model.py TICKER
```

- Trains a model from scratch using Yahoo Finance data
- Saves model, scaler, metrics, and performance plots in ./Infos/{TICKER}/

### Fine-Tune an Existing Model
```
python fine_tune.py TICKER ./Infos/{TICKER}/model_{TICKER}.keras ./Infos/{TICKER}/scaler_{TICKER}.pkl
```

- Further trains a previously saved model using a lower learning rate
- Uses early stopping and saves improved weights and plots

### Run Full Backtest
```
python complete_backtest.py TICKER [MODEL_PATH] [SCALER_PATH] [THRESHOLD]
```
Optional arguments:
- MODEL_PATH (default: ./Infos/{TICKER}/model_{TICKER}.keras)
- SCALER_PATH (default or fallback: ./Infos/scaler.pkl)
- THRESHOLD (e.g. 0.9) for fixed quantile threshold

### Simple Backtest (with custom threshold only)
```
python simple_backtest.py TICKER 0.85
```

## Outputs (per ticker)
All results are saved under the folder: ./Infos/{TICKER}/

| File                              | Description                                      |
| --------------------------------- | ------------------------------------------------ |
| `model_{TICKER}.keras`            | Trained GRU model                                |
| `scaler_{TICKER}.pkl`             | Scaler used for feature normalization            |
| `info_{TICKER}.json`              | Backtesting metrics and settings                 |
| `training_{TICKER}.png`           | Training & validation loss/MAE plot              |
| `balance_{TICKER}.png`            | Simulated equity curve                           |
| `threshold_{metric}_{TICKER}.png` | Performance by quantile (return, hit rate, etc.) |

## Model Overview
- Architecture: 3-layer GRU with normalization, Mish activations, and dropout
- Input: 90-day sequences of technical indicators and trends
- Output: Predicted next-day high return (as %)
- Loss Function: Custom Conservative Loss
  - Penalizes overestimation more than underestimation
  - Helps avoid false positives in trading decisions

## License

This project is licensed under the  
**Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)**

You may view the source, but you may not modify, redistribute, or use it commercially.

© 2025 Otero Ediciones  
[Read full license](https://creativecommons.org/licenses/by-nc-nd/4.0/)
