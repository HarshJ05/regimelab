"""Turning regime probabilities into positions, and the comparisons that keep us honest.

All rules are decided at today's close and applied to tomorrow's return.

* buy_and_hold      : always long one unit.
* hmm_filter        : long when P(stressed) < 0.5, otherwise flat (filtered, honest).
* vol_rule          : simple baseline with no HMM. Long when 20-day realised
                      volatility is at or below its median over the past year.
                      If the HMM cannot beat this, it is not adding much.
* hmm_lookahead     : same as hmm_filter but using smoothed probabilities from a
                      model fitted on the whole history. It peeks at the future,
                      so its result shows how flattering look-ahead bias is.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def apply_positions(simple_returns: pd.Series, position: pd.Series, cost_bps: float) -> pd.Series:
    pos = position.reindex(simple_returns.index).fillna(0.0).astype(float)
    held = pos.shift(1).fillna(0.0)                     # yesterday's decision
    trades = pos.diff().abs().fillna(pos.abs()).shift(1).fillna(0.0)
    return held * simple_returns - trades * cost_bps / 1e4


def vol_rule_position(log_returns_pct: pd.Series, short: int = 20, long: int = 252) -> pd.Series:
    vol = log_returns_pct.rolling(short).std()
    med = vol.rolling(long, min_periods=long // 2).median()
    pos = (vol <= med).astype(float)
    pos[vol.isna() | med.isna()] = np.nan
    return pos


def threshold_position(p_stressed: pd.Series, threshold: float = 0.5) -> pd.Series:
    pos = (p_stressed < threshold).astype(float)
    pos[p_stressed.isna()] = np.nan
    return pos
