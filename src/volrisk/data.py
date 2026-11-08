"""Data layer: download, validate, save and load raw market data."""

import pandas as pd
import yfinance as yf
from volrisk.config import START, END


def download_ticker(name, ticker, start=START, end=END):
    """Download daily OHLCV data for a single ticker from Yahoo Finance.

    Args:
        name (str): Short name of the series (e.g. "sp500"), used in error messages.
        ticker (str): Yahoo Finance ticker symbol (e.g. "^GSPC").
        start (str): First date of the sample, in 'YYYY-MM-DD' format.
        end (str): End date of the sample, in 'YYYY-MM-DD' format (exclusive).

    Returns:
        pd.DataFrame: Daily Open, High, Low, Close and Volume, indexed by date.

    Raises:
        ValueError: If the download returned no data.
    """
    # Download daily data with single-level columns
    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        multi_level_index=False,
    )

    # Raise an error if the download returned no data
    if df.empty:
        raise ValueError(
            f"No data downloaded for '{name}' ({ticker}). "
            "Check the ticker symbol, the date range and your connection."
        )

    # Remove the timezone from the index, if any
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    return df

def check_raw(df, name):
    """Check that the raw data is valid.

    Prints a short summary if all checks pass.

    Args:
        df (pd.DataFrame): Raw data to check.
        name (str): Short name of the series (e.g. "sp500"), used in error messages.

    Raises:
        ValueError: If the data is invalid.
    """
    # Check that the index is a DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"Raw data for '{name}' must have a DatetimeIndex.")

    # Check that the first date falls in the first week of the sample
    first, last = df.index.min(), df.index.max()
    start, end = pd.Timestamp(START), pd.Timestamp(END)
    week = pd.Timedelta(days=7)

    if not (start <= first < start + week):
        raise ValueError(f"Raw data for '{name}' starts on {first.date()}, expected shortly after {START}.")

    # Check that the last date falls in the last week before the (exclusive) end date
    if not (end - week <= last < end):
        raise ValueError(f"Raw data for '{name}' ends on {last.date()}, expected shortly before {END}.")

    # Check that the index is sorted in ascending order
    if not df.index.is_monotonic_increasing:
        raise ValueError(f"Raw data for '{name}' must be sorted in ascending order.")

    # Check that there are no missing values
    if df.isnull().any().any():
        raise ValueError(f"Raw data for '{name}' contains missing values.")

    # Check that there are no duplicate dates
    if df.index.duplicated().any():
        raise ValueError(f"Raw data for '{name}' contains duplicate dates in the index.")

    # Check that Open, High, Low, Close and Volume columns exist
    required_columns = ["Open", "High", "Low", "Close", "Volume"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    # Raise an error if any required columns are missing
    if missing_columns:
        raise ValueError(f"Raw data for '{name}' is missing required columns: {missing_columns}.")

    # Print a summary if all checks pass
    print(f"Raw data for '{name}' is valid: {len(df)} rows, from {first.date()} to {last.date()}.")