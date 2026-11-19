from __future__ import annotations

import numpy as np
import pandas as pd

DAYS = 252


def max_drawdown(r: pd.Series) -> float:
    eq = (1 + r.fillna(0)).cumprod()
    return float((eq / eq.cummax() - 1).min())


def summarize(r: pd.Series, position: pd.Series | None = None) -> dict:
    r = r.fillna(0.0)
    years = len(r) / DAYS
    total = float((1 + r).prod() - 1)
    cagr = float((1 + total) ** (1 / years) - 1) if years > 0 and total > -1 else np.nan
    sd = r.std(ddof=1)
    out = {
        "cagr": cagr,
        "volatility": float(sd * np.sqrt(DAYS)),
        "sharpe": float(r.mean() / sd * np.sqrt(DAYS)) if sd > 0 else np.nan,
        "max_drawdown": max_drawdown(r),
        "total_return": total,
        "years": float(years),
    }
    out["calmar"] = float(cagr / abs(out["max_drawdown"])) if out["max_drawdown"] < 0 else np.nan
    if position is not None:
        p = position.reindex(r.index).fillna(0)
        out["time_invested"] = float(p.mean())
        out["switches"] = int((p.diff().abs() > 0).sum())
    return out
