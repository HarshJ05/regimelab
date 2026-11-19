"""Walk-forward regime detection: the model only ever knows the past.

Every `refit_every` days, fit the HMM on all returns so far (expanding window,
at least `min_train` days), choose 2 or 3 regimes by BIC, then run the forward
filter through the next block. The probability for day t uses returns up to
day t only, and parameters fitted before the block started.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .hmm import HMMParams, filter_probs, select_k, smooth_probs

NAMES = {1: ["single"], 2: ["calm", "stressed"], 3: ["calm", "normal", "stressed"]}


@dataclass
class WalkForward:
    probs: pd.DataFrame                  # p_stressed, p_calm, k, label (filtered, honest)
    refits: list = field(default_factory=list)


def walk_forward(returns: pd.Series, ks=(2, 3), min_train: int = 756, refit_every: int = 126,
                 restarts: int = 4, seed: int = 0) -> WalkForward:
    r = returns.dropna()
    x = r.values
    n = len(x)
    out = pd.DataFrame(index=r.index, columns=["p_stressed", "p_calm", "k", "label"], dtype=object)
    refits, warm = [], {}
    for i in range(min_train, n, refit_every):
        best, info = select_k(x[:i], ks, restarts=restarts, seed=seed, warm=warm)
        warm = info["fits"]
        j = min(i + refit_every, n)
        a = filter_probs(x[:j], best)[i:j]
        names = NAMES[best.k]
        out.iloc[i:j, 0] = a[:, -1]                 # highest-volatility state
        out.iloc[i:j, 1] = a[:, 0]                  # lowest-volatility state
        out.iloc[i:j, 2] = best.k
        out.iloc[i:j, 3] = [names[s] for s in a.argmax(axis=1)]
        refits.append({"date": str(r.index[i].date()), "k": best.k,
                       "bic": {str(k): float(v) for k, v in info["bic"].items()},
                       "vol_annual": (np.sqrt(best.var) * np.sqrt(252)).tolist(),
                       "mean_annual": (best.mu * 252).tolist()})
    out["p_stressed"] = pd.to_numeric(out["p_stressed"])
    out["p_calm"] = pd.to_numeric(out["p_calm"])
    return WalkForward(out, refits)


@dataclass
class FullSample:
    params: HMMParams
    bic: dict
    smoothed: pd.DataFrame       # look-ahead probabilities (for description only)
    filtered: pd.DataFrame


def full_sample(returns: pd.Series, ks=(1, 2, 3), restarts: int = 6, seed: int = 0) -> FullSample:
    """Fit on the whole history. Fine for describing the past, NOT for trading."""
    r = returns.dropna()
    best, info = select_k(r.values, ks, restarts=restarts, seed=seed)
    names = NAMES[best.k]
    sm = pd.DataFrame(smooth_probs(r.values, best), index=r.index, columns=names)
    fl = pd.DataFrame(filter_probs(r.values, best), index=r.index, columns=names)
    return FullSample(best, {str(k): float(v) for k, v in info["bic"].items()}, sm, fl)


def regime_stats(returns: pd.Series, labels: pd.Series) -> list[dict]:
    """Return and volatility of the asset inside each regime (annualised)."""
    df = pd.concat([returns, labels], axis=1, keys=["r", "label"]).dropna()
    rows = []
    for lab, g in df.groupby("label"):
        r = g["r"].astype(float) / 100.0
        rows.append({"regime": lab, "share_of_days": len(g) / len(df),
                     "annual_return": float(r.mean() * 252),
                     "annual_vol": float(r.std() * np.sqrt(252))})
    return rows
