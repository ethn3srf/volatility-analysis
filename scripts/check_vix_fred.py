"""Cross-check the Yahoo Finance VIX against the official FRED series (VIXCLS).

Loads the raw VIX close from data/raw/vix.csv and the FRED VIXCLS series from
data/raw/vixcls_fred.csv, reports the values on suspect dates (US market holidays
found only in the Yahoo data), aligns both series on their common dates, and
measures the absolute differences against a fixed tolerance.

Usage:
    python scripts/check_vix_fred.py
"""

import pandas as pd

from volrisk.config import DATA_RAW
from volrisk.data import load_raw

# --- Constants -----------------------------------------------------------
FRED_FILE = DATA_RAW / "vixcls_fred.csv"  # FRED VIXCLS series
TOLERANCE = 0.01                          # maximum accepted absolute difference
SUSPECT_DATES = ["2026-05-25", "2026-09-07"]  # US market holidays found only in the Yahoo data


def load_fred_vix():
    """Load the FRED VIXCLS series from data/raw/vixcls_fred.csv.

    Returns:
        pd.Series: FRED VIX close, indexed by date, named "vix_fred".
    """
    df = pd.read_csv(FRED_FILE, index_col=0, parse_dates=True, na_values=".")
    return df["VIXCLS"].rename("vix_fred")


def report_suspect_dates(vix_yahoo, vix_fred):
    """Print the Yahoo and FRED values on each suspect date.

    Args:
        vix_yahoo (pd.Series): Yahoo VIX close.
        vix_fred (pd.Series): FRED VIX close.
    """
    print("Suspect dates (US market holidays found only in the Yahoo data):")
    for date in pd.to_datetime(SUSPECT_DATES):
        yahoo = vix_yahoo.get(date)
        fred = vix_fred.get(date)
        yahoo_str = f"{yahoo:.2f}" if pd.notna(yahoo) else "missing"
        fred_str = f"{fred:.2f}" if pd.notna(fred) else "missing"
        print(f"  {date.date()}: Yahoo = {yahoo_str}, FRED = {fred_str}")


def compare_series(vix_yahoo, vix_fred):
    """Compare the Yahoo VIX against the FRED VIX on their common dates.

    Args:
        vix_yahoo (pd.Series): Yahoo VIX close.
        vix_fred (pd.Series): FRED VIX close.

    Returns:
        pd.DataFrame: Both series and their absolute difference, on common non-missing dates.
    """
    # Align both series on their common dates and drop missing values
    df = pd.concat([vix_yahoo.rename("vix_yahoo"), vix_fred], axis=1, join="inner").dropna()
    df["abs_diff"] = (df["vix_yahoo"] - df["vix_fred"]).abs()

    # Summarize the differences
    above = df[df["abs_diff"] > TOLERANCE]
    print(f"Common dates compared: {len(df)}")
    print(f"Maximum absolute difference: {df['abs_diff'].max():.4f}")
    print(f"Days above tolerance ({TOLERANCE}): {len(above)}")

    # Show the first few mismatches, if any
    if not above.empty:
        print("First mismatches:")
        for date, row in above.head(5).iterrows():
            print(f"  {date.date()}: Yahoo = {row['vix_yahoo']:.2f}, "
                  f"FRED = {row['vix_fred']:.2f}, diff = {row['abs_diff']:.4f}")

    return df


def main():
    """Load both VIX series, report suspect dates, compare them and print a verdict."""
    vix_yahoo = load_raw("vix")["Close"]
    vix_fred = load_fred_vix()

    report_suspect_dates(vix_yahoo, vix_fred)
    print()
    df = compare_series(vix_yahoo, vix_fred)

    # Final verdict
    n_above = (df["abs_diff"] > TOLERANCE).sum()
    if n_above == 0:
        print(f"\nPASS: Yahoo VIX matches FRED VIXCLS within {TOLERANCE} on all {len(df)} common dates.")
    else:
        print(f"\nWARNING: {n_above} of {len(df)} common dates differ by more than {TOLERANCE}.")


if __name__ == "__main__":
    main()