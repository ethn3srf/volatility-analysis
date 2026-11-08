"""Return calculations: simple and log returns, annualized growth and summary statistics.

Log returns are used for statistical analysis and volatility estimation because
they are time-additive; simple returns describe realized performance. Annualized
performance is measured with the compound annual growth rate (CAGR), not by
scaling the mean daily return by 252.
"""

import numpy as np
import pandas as pd


def simple_returns(prices):
    """Compute daily simple returns from a price series.

    Args:
        prices (pd.Series): Price series, indexed by date.

    Returns:
        pd.Series: Simple returns P(t) / P(t-1) - 1, without the first (undefined) value.
    """
    return prices.pct_change().dropna().rename("simple_return")


def log_returns(prices):
    """Compute daily log returns from a price series.

    Args:
        prices (pd.Series): Price series, indexed by date.

    Returns:
        pd.Series: Log returns ln(P(t) / P(t-1)), without the first (undefined) value.
    """
    return np.log(prices).diff().dropna().rename("log_return")


def cagr(prices):
    """Compute the compound annual growth rate (CAGR) of a price series.

    The holding period is measured in calendar days divided by 365.25,
    so leap years are taken into account.

    Args:
        prices (pd.Series): Price series, indexed by date.

    Returns:
        float: CAGR, as a decimal (e.g. 0.10 for 10% per year).

    Raises:
        ValueError: If the series has fewer than two valid prices.
    """
    prices = prices.dropna()
    if len(prices) < 2:
        raise ValueError("At least two valid prices are needed to compute a CAGR.")

    years = (prices.index[-1] - prices.index[0]).days / 365.25
    return (prices.iloc[-1] / prices.iloc[0]) ** (1 / years) - 1