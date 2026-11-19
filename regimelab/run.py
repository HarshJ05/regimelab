"""Run the full study and write the dashboard data.

    python -m regimelab.run --prices data/prices.csv --out site/data
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .data import ASSETS, MAIN, load, log_returns_pct
from .metrics import summarize
from .regime import NAMES, full_sample, regime_stats, walk_forward
from .strategy import apply_positions, threshold_position, vol_rule_position

COST_BPS = 2.0
EVENTS = [
    ("Global financial crisis", "2008-09-15", "2009-03-31"),
    ("US downgrade / euro crisis", "2011-08-01", "2011-10-31"),
    ("China devaluation selloff", "2015-08-18", "2015-09-30"),
    ("Volmageddon", "2018-02-01", "2018-02-28"),
    ("COVID crash", "2020-02-20", "2020-04-30"),
    ("2022 rate shock", "2022-01-03", "2022-10-31"),
]


def _num(v, nd=5):
    if isinstance(v, (float, np.floating)):
        return None if not math.isfinite(v) else round(float(v), nd)
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


def _clean(d):
    if isinstance(d, dict):
        return {str(k): _clean(v) for k, v in d.items()}
    if isinstance(d, (list, tuple)):
        return [_clean(v) for v in d]
    if isinstance(d, np.ndarray):
        return [_clean(v) for v in d.tolist()]
    return _num(d)


def _ser(s: pd.Series, nd=4):
    return [None if (v is None or (isinstance(v, float) and not math.isfinite(v))) else round(float(v), nd)
            for v in s.values]


def analyse_asset(prices: pd.Series, name: str) -> dict:
    r = log_returns_pct(prices)
    simple = np.exp(r / 100) - 1
    wf = walk_forward(r)
    fs = full_sample(r)
    p_stress = wf.probs["p_stressed"]
    start = p_stress.first_valid_index()

    positions = {
        "buy_and_hold": pd.Series(1.0, index=r.index),
        "hmm_filter": threshold_position(p_stress),
        "vol_rule": vol_rule_position(r),
        "hmm_lookahead": threshold_position(fs.smoothed.iloc[:, -1]),
    }
    rets = {k: apply_positions(simple, p, COST_BPS) for k, p in positions.items()}
    rets = {k: v[v.index >= start] for k, v in rets.items()}
    metrics = {k: summarize(v, positions[k][positions[k].index >= start]) for k, v in rets.items()}

    sens = []
    for th in (0.3, 0.5, 0.7):
        rr = apply_positions(simple, threshold_position(p_stress, th), COST_BPS)
        sens.append({"threshold": th, **{k: summarize(rr[rr.index >= start])[k] for k in ("sharpe", "cagr", "max_drawdown")}})

    # How quickly did the honest (filtered) model flag known stress episodes?
    events = []
    for ev, a, b in EVENTS:
        seg = p_stress[(p_stress.index >= a) & (p_stress.index <= b)].dropna()
        if len(seg) == 0:
            continue
        hit = seg[seg > 0.5]
        events.append({"event": ev, "start": a, "end": b, "avg_p_stressed": float(seg.mean()),
                       "share_flagged": float((seg > 0.5).mean()),
                       "days_to_flag": int((hit.index[0] - seg.index[0]).days) if len(hit) else None,
                       "asset_return": float(np.exp(r[(r.index >= a) & (r.index <= b)].sum() / 100) - 1)})

    labels_oos = wf.probs["label"][wf.probs.index >= start]
    eq = pd.concat({k: (1 + v).cumprod() for k, v in rets.items()}, axis=1).resample("W-FRI").last().dropna()
    p = fs.params
    detail = {
        "ticker": prices.name, "name": name,
        "dates": [d.strftime("%Y-%m-%d") for d in r.index],
        "price": _ser(prices.reindex(r.index), 2),
        "p_stressed": _ser(p_stress, 3),
        "p_stressed_smoothed": _ser(fs.smoothed.iloc[:, -1], 3),
        "label": [None if (isinstance(x, float) and np.isnan(x)) or x is None else x for x in wf.probs["label"].tolist()],
        "equity": {"dates": [d.strftime("%Y-%m-%d") for d in eq.index], **{k: _ser(eq[k]) for k in eq.columns}},
        "refits": wf.refits,
        "events": events,
    }
    summary = {
        "ticker": prices.name, "name": name, "out_of_sample_start": str(start.date()),
        "metrics": metrics, "threshold_sensitivity": sens,
        "full_sample": {
            "k": p.k, "names": NAMES[p.k], "bic": fs.bic,
            "transition": p.A.tolist(),
            "annual_vol": (np.sqrt(p.var) * np.sqrt(252) / 100).tolist(),
            "annual_mean": (p.mu * 252 / 100).tolist(),
            "expected_duration_days": p.expected_duration().tolist(),
        },
        "walk_forward_k": {str(k): sum(1 for f in wf.refits if f["k"] == k) for k in (2, 3)},
        "regime_stats_oos": regime_stats(r[r.index >= start], labels_oos),
        "events": events,
    }
    return {"summary": summary, "detail": detail, "labels": wf.probs["label"], "returns": r}


def spreadtrader_by_regime(spread_prices_path: Path, es_labels: pd.Series, es_returns: pd.Series) -> dict | None:
    """Does the spread engine behave differently when the stock market is stressed?"""
    try:
        from spreadtrader.backtest import Config
        from spreadtrader.data import clean, load as st_load
        from spreadtrader.run import first_trade_date, portfolio, run_all
    except ImportError:
        return None
    if not Path(spread_prices_path).exists():
        return None
    prices, _ = clean(st_load(spread_prices_path))
    res = run_all(prices, Config())
    port = portfolio(res, "kalman")
    port = port[port.index >= first_trade_date(res)]
    # Use yesterday's regime so the label is known before the day's return.
    lab = es_labels.shift(1).reindex(port.index)
    es = (np.exp(es_returns / 100) - 1).reindex(port.index)
    rows = []
    for name in ("calm", "normal", "stressed"):
        m = lab == name
        if m.sum() < 20:
            continue
        s, e = port[m], es[m]
        rows.append({"regime": name, "share_of_days": float(m.mean()),
                     "spread_annual_return": float(s.mean() * 252),
                     "spread_sharpe": float(s.mean() / s.std() * np.sqrt(252)) if s.std() > 0 else None,
                     "sp500_annual_return": float(e.mean() * 252),
                     "sp500_sharpe": float(e.mean() / e.std() * np.sqrt(252)) if e.std() > 0 else None})
    return {"start": str(port.index[0].date()), "rows": rows}


def report(summary: dict) -> str:
    pct = lambda v: "n/a" if v is None else f"{v*100:.1f}%"
    num = lambda v: "n/a" if v is None else f"{v:.2f}"
    L = ["# RegimeLab results", "", f"Generated {summary['generated_at']}.", ""]
    names = {"buy_and_hold": "Buy & hold", "hmm_filter": "HMM regime filter (honest)",
             "vol_rule": "Simple volatility rule", "hmm_lookahead": "HMM with look-ahead (cheating)"}
    for a in summary["assets"]:
        L += [f"## {a['name']} ({a['ticker']}), out-of-sample from {a['out_of_sample_start']}", "",
              "| Strategy | CAGR | Volatility | Sharpe | Max drawdown | Time invested |", "|---|---|---|---|---|---|"]
        for k, lab in names.items():
            m = a["metrics"][k]
            L.append(f"| {lab} | {pct(m['cagr'])} | {pct(m['volatility'])} | {num(m['sharpe'])} | "
                     f"{pct(m['max_drawdown'])} | {pct(m.get('time_invested'))} |")
        fs = a["full_sample"]
        L += ["", f"Full-sample model: {fs['k']} regimes ({', '.join(fs['names'])}); annual vol "
              + ", ".join(f"{n} {v*100:.0f}%" for n, v in zip(fs["names"], fs["annual_vol"]))
              + "; average duration " + ", ".join(f"{n} {d:.0f} days" for n, d in zip(fs["names"], fs["expected_duration_days"])), ""]
    es = summary["assets"][0]
    L += ["## Did the honest model flag known stress episodes? (S&P 500)", "",
          "| Episode | Avg P(stressed) | Days flagged | Days until first flag | S&P 500 return |", "|---|---|---|---|---|"]
    for e in es["events"]:
        L.append(f"| {e['event']} | {num(e['avg_p_stressed'])} | {pct(e['share_flagged'])} | "
                 f"{e['days_to_flag'] if e['days_to_flag'] is not None else 'never'} | {pct(e['asset_return'])} |")
    st = summary.get("spreadtrader_by_regime")
    if st:
        L += ["", f"## SpreadTrader portfolio by S&P 500 regime (from {st['start']})", "",
              "| Regime | Share of days | Spread engine annual return | Spread engine Sharpe | S&P 500 Sharpe |", "|---|---|---|---|---|"]
        for r in st["rows"]:
            L.append(f"| {r['regime']} | {pct(r['share_of_days'])} | {pct(r['spread_annual_return'])} | "
                     f"{num(r['spread_sharpe'])} | {num(r['sp500_sharpe'])} |")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", default="data/prices.csv")
    ap.add_argument("--spread-prices", default="data/spread_prices.csv")
    ap.add_argument("--out", default="site/data")
    ap.add_argument("--report", default="results/RESULTS.md")
    a = ap.parse_args()
    prices = load(a.prices)
    out = Path(a.out)
    (out / "assets").mkdir(parents=True, exist_ok=True)
    assets, main_res = [], None
    for t, name in ASSETS.items():
        res = analyse_asset(prices[t].rename(t), name)
        (out / "assets" / f"{t.replace('=F', '')}.json").write_text(json.dumps(_clean(res["detail"])))
        assets.append(res["summary"])
        if t == MAIN:
            main_res = res
        print(f"done {t}")
    summary = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "data": {"first_date": str(prices.index.min().date()), "last_date": str(prices.index.max().date())},
        "cost_bps": COST_BPS,
        "assets": assets,
        "spreadtrader_by_regime": spreadtrader_by_regime(Path(a.spread_prices), main_res["labels"], main_res["returns"]),
    }
    summary = _clean(summary)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    Path(a.report).write_text(report(summary))
    print(report(summary))


if __name__ == "__main__":
    main()
