#!/usr/bin/env python3
from pathlib import Path
import json
import shutil

BASE_DIR = Path(__file__).resolve().parent
OTHER_INFOS = BASE_DIR / "OtherInfos"
FINALISTS_DIR = BASE_DIR / "Finalists"
THRESHOLDS_JSON = FINALISTS_DIR / "tickers.json"

def find_ticker_dirs(root: Path):
    # Expect structure: /Training/OtherInfos/{training_batch}/{ticker}
    # Only include directories that look like a ticker folder (contain info_*.json)
    for batch_dir in sorted((root).glob("*")):
        if not batch_dir.is_dir():
            continue
        for ticker_dir in sorted(batch_dir.glob("*")):
            if not ticker_dir.is_dir():
                continue
            # Quick sanity check: must have an info_*.json file
            info_files = list(ticker_dir.glob("info_*.json"))
            if info_files:
                yield ticker_dir

def load_info(ticker_dir: Path):
    # There should be exactly one info_*.json – if multiple, take the first deterministically
    info_path = sorted(ticker_dir.glob("info_*.json"))[0]
    with info_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    # Required fields
    wr = float(data.get("weighted_return", float("-inf")))
    thr = data.get("threshold", None)
    return wr, thr

def find_model_and_scaler(ticker_dir: Path, ticker: str):
    # Prefer exact names; if not found, fallback to any model_*.keras / scaler_*.pkl
    model_path = ticker_dir / f"model_{ticker}.keras"
    scaler_path = ticker_dir / f"scaler_{ticker}.pkl"
    if not model_path.exists():
        candidates = list(ticker_dir.glob("model_*.keras"))
        model_path = candidates[0] if candidates else None
    if not scaler_path.exists():
        candidates = list(ticker_dir.glob("scaler_*.pkl"))
        scaler_path = candidates[0] if candidates else None
    return model_path, scaler_path

def main():
    if not OTHER_INFOS.exists():
        raise SystemExit(f"Not found: {OTHER_INFOS}")
    FINALISTS_DIR.mkdir(parents=True, exist_ok=True)

    best_by_ticker = {}  # ticker -> dict(dir, weighted_return, threshold, model_path, scaler_path)

    for ticker_dir in find_ticker_dirs(OTHER_INFOS):
        ticker = ticker_dir.name  # use folder name as the canonical ticker
        try:
            weighted_return, threshold = load_info(ticker_dir)
        except Exception as e:
            print(f"[WARN] Skipping {ticker_dir} (bad info json): {e}")
            continue

        model_path, scaler_path = find_model_and_scaler(ticker_dir, ticker)
        if not model_path or not model_path.exists():
            print(f"[WARN] Missing model for {ticker} in {ticker_dir}")
            continue
        if not scaler_path or not scaler_path.exists():
            print(f"[WARN] Missing scaler for {ticker} in {ticker_dir}")
            continue

        current = best_by_ticker.get(ticker)
        if (current is None) or (weighted_return > current["weighted_return"]):
            best_by_ticker[ticker] = {
                "dir": ticker_dir,
                "weighted_return": weighted_return,
                "threshold": threshold,
                "model_path": model_path,
                "scaler_path": scaler_path,
            }

    # Copy winners to /Training/Finalists and build thresholds map
    thresholds_map = {}
    for ticker, info in sorted(best_by_ticker.items()):
        # Copy files with standardized names
        dst_model = FINALISTS_DIR / f"model_{ticker}.keras"
        dst_scaler = FINALISTS_DIR / f"scaler_{ticker}.pkl"
        shutil.copy2(info["model_path"], dst_model)
        shutil.copy2(info["scaler_path"], dst_scaler)
        thresholds_map[ticker] = {"threshold": info["threshold"]}

        print(f"[OK] {ticker}: weighted_return={info['weighted_return']:.6f} "
              f"-> copied model and scaler")

    # Write thresholds.json
    with THRESHOLDS_JSON.open("w", encoding="utf-8") as f:
        json.dump(thresholds_map, f, ensure_ascii=False, indent=2)

    print(f"\nWrote thresholds for {len(thresholds_map)} tickers to {THRESHOLDS_JSON}")
    print(f"Models and scalers are in {FINALISTS_DIR}")

if __name__ == "__main__":
    main()
