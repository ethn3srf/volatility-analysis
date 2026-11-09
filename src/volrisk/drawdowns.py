"""Drawdown analysis: drawdown series, maximum drawdown and drawdown episodes.

The drawdown at date t is the percentage loss from the highest price reached up to t:
DD_t = P_t / max(P_s, s <= t) - 1. Episodes are classified with standard market conventions,
fixed before looking at the results: a correction is a drawdown of at least 10%, and a bear market
a drawdown of at least 20%. An episode not yet recovered at the end of the sample has a
right-censored duration: it is known to last at least until the last observation.
"""

import numpy as np
import pandas as pd


def drawdown_series(prices):
    """Compute the drawdown of a price series relative to its running maximum.

    Args:
        prices (pd.Series): Price series, indexed by date.

    Returns:
        pd.Series: Drawdown at each date, as a decimal (0 at a new high, negative below it).
    """
    prices = prices.dropna()
    return (prices / prices.cummax() - 1).rename("drawdown")


def max_drawdown(prices):
    """Return the maximum drawdown (the deepest peak-to-trough loss) of a price series.

    Args:
        prices (pd.Series): Price series, indexed by date.

    Returns:
        float: Maximum drawdown, as a negative decimal (e.g. -0.34 for -34%).
    """
    return drawdown_series(prices).min()


def drawdown_episodes(prices, min_depth=0.10):
    """List every drawdown episode at least as deep as min_depth.

    An episode starts at a price peak, reaches a trough, and ends when the price recovers its
    previous peak. Durations are measured in trading days.

    Args:
        prices (pd.Series): Price series, indexed by date.
        min_depth (float): Minimum depth to keep an episode (0.10 = corrections of 10% or more).

    Returns:
        pd.DataFrame: One row per episode with peak, trough and recovery dates, depth,
            durations, and a flag for episodes not yet recovered (right-censored).
    """
    dd = drawdown_series(prices)
    position = pd.Series(np.arange(len(dd)), index=dd.index)  # trading-day position of each date
    episode_id = (dd == 0).cumsum()  # a new id starts at each new high

    rows = []
    for eid, segment in dd.groupby(episode_id):
        depth = segment.min()
        if depth > -min_depth:
            continue  # too shallow: not a correction

        peak, trough = segment.index[0], segment.idxmin()
        recovered = (episode_id == eid + 1).any()
        recovery = (episode_id == eid + 1).idxmax() if recovered else pd.NaT

        rows.append({
            "peak": peak.date(),
            "trough": trough.date(),
            "recovery": recovery.date() if recovered else None,
            "depth": depth,
            "days_to_trough": position[trough] - position[peak],
            "days_to_recovery": position[recovery] - position[trough] if recovered else np.nan,
            "total_days": (position[recovery] if recovered else position.iloc[-1]) - position[peak],
            "recovered": recovered,
        })

    return pd.DataFrame(rows).sort_values("depth").reset_index(drop=True)