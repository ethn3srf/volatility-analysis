"""Market risk measures: Value-at-Risk, Expected Shortfall and VaR backtesting.

Losses are reported as positive numbers: a 1-day 99% VaR of 0.03 means that the daily loss
is expected to exceed 3% on 1% of days. Expected Shortfall (ES) is the average loss on those
days; unlike VaR, it is a coherent risk measure (Artzner, Delbaen, Eber & Heath, 1999).
VaR forecasts are backtested with the unconditional coverage test of Kupiec (1995).
"""

import pandas as pd
from scipy import stats
from scipy.special import xlogy


def historical_var(returns, level=0.99):
    """Historical VaR: the loss at the (1 - level) quantile of past returns.

    Args:
        returns (pd.Series): Daily returns, indexed by date.
        level (float): Confidence level (e.g. 0.95 or 0.99).

    Returns:
        float: VaR, as a positive loss.
    """
    return -returns.dropna().quantile(1 - level)


def historical_es(returns, level=0.99):
    """Historical Expected Shortfall: the average loss beyond the historical VaR.

    Args:
        returns (pd.Series): Daily returns, indexed by date.
        level (float): Confidence level.

    Returns:
        float: Expected Shortfall, as a positive loss.
    """
    r = returns.dropna()
    threshold = r.quantile(1 - level)
    return -r[r <= threshold].mean()


def normal_var(returns, level=0.99):
    """Parametric VaR assuming normally distributed returns with the sample mean and volatility.

    Args:
        returns (pd.Series): Daily returns, indexed by date.
        level (float): Confidence level.

    Returns:
        float: VaR, as a positive loss.
    """
    r = returns.dropna()
    z = stats.norm.ppf(1 - level)
    return -(r.mean() + z * r.std())


def normal_es(returns, level=0.99):
    """Parametric Expected Shortfall under the normal distribution.

    ES = -mu + sigma * phi(z) / (1 - level), where z is the (1 - level) standard normal quantile.

    Args:
        returns (pd.Series): Daily returns, indexed by date.
        level (float): Confidence level.

    Returns:
        float: Expected Shortfall, as a positive loss.
    """
    r = returns.dropna()
    z = stats.norm.ppf(1 - level)
    return -r.mean() + r.std() * stats.norm.pdf(z) / (1 - level)


def rolling_var(returns, level=0.99, window=252, method="historical"):
    """One-day-ahead VaR forecasts estimated on a rolling window (no look-ahead).

    The forecast for day t uses only the `window` returns up to day t-1.

    Args:
        returns (pd.Series): Daily returns, indexed by date.
        level (float): Confidence level.
        window (int): Estimation window, in trading days (252 = one year).
        method (str): "historical" or "normal".

    Returns:
        pd.Series: VaR forecast for each day, as a positive loss.

    Raises:
        ValueError: If the method is not recognized.
    """
    rolling = returns.rolling(window)
    if method == "historical":
        var = -rolling.quantile(1 - level)
    elif method == "normal":
        var = -(rolling.mean() + stats.norm.ppf(1 - level) * rolling.std())
    else:
        raise ValueError(f"Unknown method '{method}': use 'historical' or 'normal'.")
    return var.shift(1).dropna().rename(f"var_{method}_{int(level * 100)}")


def kupiec_test(returns, var_forecast, level=0.99):
    """Kupiec (1995) proportion-of-failures test of a VaR backtest.

    An exceedance occurs when the realized loss is larger than the VaR forecast. Under a correct
    model, the exceedance rate equals 1 - level. The likelihood-ratio statistic is
    asymptotically chi-squared with one degree of freedom.

    Args:
        returns (pd.Series): Realized daily returns.
        var_forecast (pd.Series): VaR forecasts (positive losses), aligned on dates.
        level (float): Confidence level of the VaR.

    Returns:
        pd.Series: Number of days, exceedances, observed and expected rates, LR statistic and p-value.
    """
    data = pd.concat([returns.rename("ret"), var_forecast.rename("var")], axis=1, join="inner").dropna()
    exceed = data["ret"] < -data["var"]
    n, x = len(exceed), int(exceed.sum())
    p, p_hat = 1 - level, x / n

    log_lik_null = xlogy(n - x, 1 - p) + xlogy(x, p)
    log_lik_alt = xlogy(n - x, 1 - p_hat) + xlogy(x, p_hat)
    lr = -2 * (log_lik_null - log_lik_alt)

    return pd.Series({
        "n_days": n,
        "exceedances": x,
        "observed_rate": p_hat,
        "expected_rate": p,
        "lr_stat": lr,
        "p_value": stats.chi2.sf(lr, df=1),
    }, name=var_forecast.name)