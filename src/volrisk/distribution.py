"""Distribution diagnostics: normality tests, autocorrelation and volatility-standardized returns.

These tools test the stylized facts of asset returns (Cont, 2001): non-normality
(Jarque & Bera, 1980), absence of autocorrelation in returns but strong autocorrelation
in absolute returns, i.e. volatility clustering (Engle, 1982), tested with the
Ljung-Box statistic (Ljung & Box, 1978).
"""

import numpy as np
import pandas as pd
from scipy import stats


def normality_test(returns):
    """Run the Jarque-Bera normality test on a return series.

    Args:
        returns (pd.Series): Return series, indexed by date.

    Returns:
        pd.Series: Number of observations, skewness, excess kurtosis, Jarque-Bera statistic and p-value.
    """
    r = returns.dropna()
    jb_stat, p_value = stats.jarque_bera(r)
    return pd.Series({
        "n_obs": r.count(),
        "skewness": r.skew(),
        "excess_kurtosis": r.kurt(),
        "jarque_bera": jb_stat,
        "p_value": p_value,
    }, name=returns.name)


def autocorrelation(series, max_lag=20):
    """Compute the sample autocorrelation of a series for lags 1 to max_lag.

    Args:
        series (pd.Series): Series indexed by date (e.g. returns or absolute returns).
        max_lag (int): Largest lag, in trading days.

    Returns:
        pd.Series: Autocorrelation at each lag, indexed by lag.
    """
    s = series.dropna()
    lags = range(1, max_lag + 1)
    return pd.Series([s.autocorr(lag) for lag in lags],
                     index=pd.Index(lags, name="lag"), name=series.name)


def ljung_box(series, max_lag=20):
    """Ljung-Box test of joint zero autocorrelation up to max_lag.

    Q = n (n + 2) * sum_{k=1..m} rho_k^2 / (n - k), asymptotically chi-squared with m degrees of freedom.

    Args:
        series (pd.Series): Series indexed by date.
        max_lag (int): Number of lags tested (m).

    Returns:
        pd.Series: Q statistic and p-value.
    """
    n = series.dropna().count()
    rho = autocorrelation(series, max_lag)
    q_stat = n * (n + 2) * sum(rho[k] ** 2 / (n - k) for k in rho.index)
    p_value = stats.chi2.sf(q_stat, df=max_lag)
    return pd.Series({"ljung_box_q": q_stat, "p_value": p_value}, name=series.name)


def standardized_returns(returns, window=21):
    """Divide each return by the volatility estimated over the previous window (no look-ahead).

    z_t = r_t / sigma_(t-1), where sigma_(t-1) is the standard deviation of the `window`
    returns up to and including day t-1.

    Args:
        returns (pd.Series): Daily log returns, indexed by date.
        window (int): Rolling window used to estimate volatility, in trading days.

    Returns:
        pd.Series: Volatility-standardized returns.
    """
    sigma = returns.rolling(window).std().shift(1)
    return (returns / sigma).dropna().rename(f"std_return_{window}d")