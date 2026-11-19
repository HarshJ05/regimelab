"""Download and clean daily futures prices (Yahoo Finance continuous front month)."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ASSETS = {
    "ES=F": "S&P 500 E-mini",
    "ZN=F": "US 10-year Treasury note",
    "GC=F": "Gold",
    "CL=F": "WTI crude oil",
}
MAIN = "ES=F"
DEFAULT_PATH = Path("data/prices.csv")


def _download_one(yf, ticker, start, end=None, attempts=4) -> pd.Series:
    import time
    for i in range(attempts):
        raw = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False, threads=False)
        if raw is not None and len(raw):
            close = raw["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            return close.rename(ticker)
        time.sleep(3 * (i + 1))
    return pd.Series(name=ticker, dtype=float)


def download(tickers, start="2000-01-01", end=None) -> pd.DataFrame:
    import yfinance as yf
    df = pd.concat([_download_one(yf, t, start, end) for t in tickers], axis=1)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "date"
    return df.sort_index()


def log_returns_pct(prices: pd.Series) -> pd.Series:
    """Daily log returns in percent. Non-positive prices (WTI on 20 Apr 2020) are dropped first."""
    p = prices.where(prices > 0).dropna()
    return (100 * np.log(p).diff()).dropna()


def load(path=DEFAULT_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "date"
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2000-01-01")
    ap.add_argument("--out", default=str(DEFAULT_PATH))
    ap.add_argument("--spreadtrader-out", default="data/spread_prices.csv",
                    help="also download the SpreadTrader universe if that package is installed")
    a = ap.parse_args()
    df = download(list(ASSETS), a.start)
    missing = [t for t in ASSETS if t not in df or df[t].notna().sum() == 0]
    if missing:
        raise SystemExit(f"No data for {missing}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, float_format="%.6f")
    print(f"Saved {df.shape} to {a.out}")
    try:
        from spreadtrader.universe import TICKERS
    except ImportError:
        print("spreadtrader not installed; skipping its prices")
        return
    sp = download(TICKERS, "2010-01-01")
    sp.to_csv(a.spreadtrader_out, float_format="%.6f")
    print(f"Saved SpreadTrader prices {sp.shape} to {a.spreadtrader_out}")


if __name__ == "__main__":
    main()
