"""Implied vs. realized volatility: does the VIX forecast future S&P 500 volatility?

The VIX at date t is compared with the volatility realized over the next 21 trading days
(t+1 to t+21), and with a naive forecast (the volatility realized over the previous 21 days).
Forecast quality is assessed with Mincer & Zarnowitz (1969) regressions, using Newey & West (1987)
standard errors because 21-day windows overlap. The variance risk premium VIX^2 - RV^2 follows
Carr & Wu (2009) and Bollerslev, Tauchen & Zhou (2009).
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

from volrisk.config import TRADING_DAYS


def realized_future_vol(returns, horizon=21):
    """Annualized volatility realized over the next `horizon` trading days (t+1 to t+horizon).

    Args:
        returns (pd.Series): Daily log returns, indexed by date.
        horizon (int): Forecast horizon, in trading days (21 = about 30 calendar days, as for the VIX).

    Returns:
        pd.Series: Realized future volatility; the last `horizon` dates are dropped (future not yet known).
    """
    rv = returns.rolling(horizon).std().shift(-horizon) * np.sqrt(TRADING_DAYS)
    return rv.dropna().rename("realized_vol")


def build_forecast_table(vix, returns, horizon=21):
    """Align the VIX, the naive forecast and realized future volatility on common dates.

    Args:
        vix (pd.Series): VIX in decimal form, indexed by date.
        returns (pd.Series): Daily log returns of the S&P 500, indexed by date.
        horizon (int): Forecast horizon, in trading days.

    Returns:
        pd.DataFrame: Columns vix, naive (past `horizon`-day volatility) and realized_vol.
    """
    naive = (returns.rolling(horizon).std() * np.sqrt(TRADING_DAYS)).rename("naive")
    table = pd.concat([vix, naive, realized_future_vol(returns, horizon)], axis=1, join="inner").dropna()

    dropped = vix.index.difference(table.index).intersection(vix.index[vix.index < table.index[-1]])
    if len(dropped) > 0:
        print(f"{len(dropped)} VIX dates without matching S&P 500 data were dropped "
              f"(e.g. {', '.join(str(d.date()) for d in dropped[:5])}).")
    return table


def forecast_errors(table, target="realized_vol", forecasts=("vix", "naive")):
    """Summarize forecast errors: mean bias, mean absolute error and root mean squared error.

    Args:
        table (pd.DataFrame): Output of build_forecast_table.
        target (str): Column with the realized value.
        forecasts (tuple): Columns with the forecasts to evaluate.

    Returns:
        pd.DataFrame: One column per forecast; rows bias (forecast - realized), MAE and RMSE.
    """
    out = {}
    for f in forecasts:
        err = table[f] - table[target]
        out[f] = {"bias": err.mean(), "mae": err.abs().mean(), "rmse": np.sqrt((err ** 2).mean())}
    return pd.DataFrame(out)


def mincer_zarnowitz(table, forecast, target="realized_vol", hac_lags=21):
    """Mincer-Zarnowitz regression: realized = a + b * forecast + error, with Newey-West standard errors.

    An unbiased, efficient forecast has a = 0 and b = 1. The joint hypothesis is tested with a Wald test.

    Args:
        table (pd.DataFrame): Output of build_forecast_table.
        forecast (str): Column used as the forecast (e.g. "vix" or "naive").
        target (str): Column with realized volatility.
        hac_lags (int): Lags for the Newey-West covariance (21 = the overlap of the windows).

    Returns:
        pd.Series: Intercept a, slope b, their standard errors, R-squared and the p-value of the
            joint test a = 0 and b = 1.
    """
    X = sm.add_constant(table[[forecast]])
    res = sm.OLS(table[target], X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    wald = res.wald_test(f"const = 0, {forecast} = 1", scalar=True)
    return pd.Series({
        "a": res.params["const"],
        "b": res.params[forecast],
        "se_a": res.bse["const"],
        "se_b": res.bse[forecast],
        "r_squared": res.rsquared,
        "p_value_a0_b1": float(wald.pvalue),
        "n_obs": int(res.nobs),
    }, name=forecast)


def variance_risk_premium(table):
    """Variance risk premium: implied variance (VIX^2) minus realized variance, both annualized.

    Args:
        table (pd.DataFrame): Output of build_forecast_table.

    Returns:
        pd.Series: VIX^2 - RV^2 at each date; positive values mean the VIX overstated future variance.
    """
    return (table["vix"] ** 2 - table["realized_vol"] ** 2).rename("vrp")