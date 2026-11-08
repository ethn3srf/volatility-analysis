"""Data layer: download, validate, save and load raw market data."""

import pandas as pd
import yfinance as yf
from volrisk.config import DATA_RAW, START, END


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

def save_raw(df, name, force=False):
    """Save raw data to a CSV file.

    Args:
        df (pd.DataFrame): Raw data to save.
        name (str): Short name of the series (e.g. "sp500"), used in the filename.
        force (bool): If True, overwrite existing file. Default is False.

    Raises:
        FileExistsError: If the file already exists and force is False. Existing files are never overwritten unless force=True, so the original download stays frozen.
    """
    # Construct the filename
    filename = f"{name}.csv"
    filepath = DATA_RAW / filename

    # Check if the file already exists
    if filepath.exists() and not force:
        raise FileExistsError(f"File '{filepath}' already exists. Use force=True to overwrite.")

    # Save the DataFrame to CSV
    df.to_csv(filepath)
    print(f"Raw data for '{name}' saved to '{filepath}'.")

def save_raw(df, name, force=False):
    """Save raw data to a CSV file.

    Existing files are never overwritten unless force=True, so the original download stays frozen.

    Args:
        df (pd.DataFrame): Raw data to save.
        name (str): Short name of the series (e.g. "sp500"), used in the filename.
        force (bool): If True, overwrite an existing file. Default is False.

    Raises:
        FileExistsError: If the file already exists and force is False.
    """
    # Build the file path
    filepath = DATA_RAW / f"{name}.csv"

    # Refuse to overwrite an existing raw file unless explicitly forced
    if filepath.exists() and not force:
        raise FileExistsError(f"File '{filepath}' already exists. Use force=True to overwrite.")

    # Create the raw data folder if needed, then save the DataFrame to CSV
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath)
    print(f"Raw data for '{name}' saved to '{filepath}'.")


def load_raw(name):
    """Load raw data from a CSV file.

    Args:
        name (str): Short name of the series (e.g. "sp500"), used in the filename.

    Returns:
        pd.DataFrame: Raw data indexed by date.

    Raises:
        FileNotFoundError: If the raw file does not exist yet.
    """
    # Build the file path
    filepath = DATA_RAW / f"{name}.csv"

    # Stop with a clear message if the raw data has not been downloaded yet
    if not filepath.exists():
        raise FileNotFoundError(
            f"Raw data for '{name}' not found at '{filepath}'. Run scripts/download_data.py first."
        )

    # Load the DataFrame from CSV, with dates as index
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    print(f"Raw data for '{name}' loaded from '{filepath}'.")
    return df