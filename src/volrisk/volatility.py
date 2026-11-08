"""Volatility estimators: rolling close-to-close, Parkinson and Garman-Klass.

All estimators return annualized volatility, using the square-root-of-time rule
with TRADING_DAYS trading days per year. The close-to-close estimator uses daily
log returns; the range-based estimators (Parkinson, 1980; Garman & Klass, 1980)
use daily open, high, low and close prices. Range-based estimators ignore the
overnight gap between the previous close and the open, so they tend to understate
volatility on days with large opening jumps.
"""

import numpy as np
import pandas as pd

from volrisk.config import TRADING_DAYS


def rolling_volatility(returns, window):
    """Compute annualized rolling close-to-close volatility.

    Args:
        returns (pd.Series): Daily log returns, indexed by date.
        window (int): Rolling window length, in trading days (e.g. 21 or 63).

    Returns:
        pd.Series: Annualized rolling volatility, without the first window - 1 (undefined) values.
    """
    vol = returns.rolling(window).std() * np.sqrt(TRADING_DAYS)
    return vol.dropna().rename(f"vol_cc_{window}d")


def parkinson_volatility(high, low, window):
    """Compute annualized rolling Parkinson (1980) volatility from daily high and low prices.

    The daily variance is estimated as ln(H / L)^2 / (4 ln 2), averaged over the window.

    Args:
        high (pd.Series): Daily high prices, indexed by date.
        low (pd.Series): Daily low prices, indexed by date.
        window (int): Rolling window length, in trading days.

    Returns:
        pd.Series: Annualized rolling Parkinson volatility.
    """
    daily_var = np.log(high / low) ** 2 / (4 * np.log(2))
    vol = np.sqrt(daily_var.rolling(window).mean() * TRADING_DAYS)
    return vol.dropna().rename(f"vol_parkinson_{window}d")


def garman_klass_volatility(open_, high, low, close, window):
    """Compute annualized rolling Garman-Klass (1980) volatility from daily OHLC prices.

    The daily variance is estimated as 0.5 ln(H / L)^2 - (2 ln 2 - 1) ln(C / O)^2,
    averaged over the window.

    Args:
        open_ (pd.Series): Daily open prices, indexed by date.
        high (pd.Series): Daily high prices, indexed by date.
        low (pd.Series): Daily low prices, indexed by date.
        close (pd.Series): Daily close prices, indexed by date.
        window (int): Rolling window length, in trading days.

    Returns:
        pd.Series: Annualized rolling Garman-Klass volatility.
    """
    daily_var = 0.5 * np.log(high / low) ** 2 - (2 * np.log(2) - 1) * np.log(close / open_) ** 2
    vol = np.sqrt(daily_var.rolling(window).mean() * TRADING_DAYS)
    return vol.dropna().rename(f"vol_gk_{window}d")