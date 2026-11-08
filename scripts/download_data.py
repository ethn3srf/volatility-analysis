"""Download raw daily market data and freeze it in data/raw.

Fetches daily OHLCV data for every ticker in config.TICKERS (S&P 500 and VIX)
from Yahoo Finance over the sample period [START, END), validates each series,
and saves it as a CSV file in data/raw. Existing files are never overwritten.
A metadata.txt file records the download date, the yfinance version and the
sample period, so the dataset can be traced and reproduced.

Usage:
    python scripts/download_data.py
"""

from datetime import datetime

import yfinance as yf

from volrisk.config import DATA_RAW, END, START, TICKERS
from volrisk.data import check_raw, download_ticker, save_raw


def main():
    """Download, validate and save every series, then write the metadata file."""
    # Download, validate and save each series
    for name, ticker in TICKERS.items():
        df = download_ticker(name, ticker)
        check_raw(df, name)
        save_raw(df, name)

    # Record how and when the raw data was obtained (written only if every series succeeded)
    metadata = [
        f"download_date: {datetime.now().isoformat(timespec='seconds')}",
        f"source: Yahoo Finance (yfinance {yf.__version__})",
        f"sample_period: [{START}, {END})",
        f"tickers: {TICKERS}",
    ]
    metadata_path = DATA_RAW / "metadata.txt"
    metadata_path.write_text("\n".join(metadata) + "\n", encoding="utf-8")
    print(f"Metadata saved to '{metadata_path}'.")


if __name__ == "__main__":
    main()