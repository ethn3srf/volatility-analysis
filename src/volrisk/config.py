"""Central configuration: project paths, tickers, sample period and analysis parameters."""

from pathlib import Path

# --- Paths ---------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "figures"

# --- Data ----------------------------------------------------------------
TICKERS = {
    "sp500": "^GSPC",
    "vix": "^VIX",
}
START = "2015-01-01"
END = "2026-11-01"  # yfinance treats the end date as exclusive: sample ends on 2026-10-31

# --- Analysis parameters -------------------------------------------------
WINDOWS = [21, 63]                # rolling windows: 1 trading month and 1 trading quarter
CONFIDENCE_LEVELS = [0.95, 0.99]  # confidence levels for VaR and Expected Shortfall
TRADING_DAYS = 252                # trading days per year, used to annualize volatility