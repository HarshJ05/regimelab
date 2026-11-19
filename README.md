# RegimeLab: market regimes with a from-scratch Hidden Markov Model

[![tests](https://github.com/HarshJ05/regimelab/actions/workflows/tests.yml/badge.svg)](https://github.com/HarshJ05/regimelab/actions/workflows/tests.yml)
[![backtest](https://github.com/HarshJ05/regimelab/actions/workflows/backtest.yml/badge.svg)](https://github.com/HarshJ05/regimelab/actions/workflows/backtest.yml)

**Live dashboard:** https://harshj05.github.io/regimelab/

Markets switch between calm and stressed periods. RegimeLab fits a **Gaussian Hidden Markov Model, written from
scratch** (Baum-Welch EM, forward filter, smoother, BIC model selection, numba-accelerated), to daily returns of four
futures markets: S&P 500, 10-year Treasuries, gold and WTI crude. It then tests whether stepping out of the market in
stressed regimes helps, **walk-forward and without look-ahead**.

- **Walk-forward:** refit every 6 months on all data so far; BIC picks 2 or 3 regimes each time. The probability for
  day t uses returns up to day t only.
- **Filtered vs smoothed:** the project computes both and shows how much a backtest is inflated if you accidentally
  use smoothed (hindsight) probabilities.
- **Honest baselines:** buy & hold, and a simple rule with no HMM (flat when 20-day volatility is above its 1-year median).
- **Crisis check:** how quickly the honest model flagged 2008, 2011, 2015, 2018, COVID and the 2022 rate shock.
- **Linked to [SpreadTrader](https://github.com/HarshJ05/spreadtrader):** the futures spread portfolio, split by S&P 500 regime.
- **Self-updating:** a weekly GitHub Action downloads prices, re-fits everything and redeploys the dashboard.

## Results

Daily data from 2000 to Nov 2026; trading out-of-sample from Sept 2003. Costs 2 bp per switch.
Full tables: [results/RESULTS.md](results/RESULTS.md).

**S&P 500 E-mini futures**

| Strategy | Sharpe | Annual return | Max drawdown | Time invested |
|---|---|---|---|---|
| Buy & hold | 0.55 | 8.9% | −57.1% | 100% |
| **HMM regime filter (honest)** | **0.67** | 9.0% | **−27.3%** | 94% |
| Simple volatility rule | 0.41 | 3.4% | −21.7% | 53% |
| HMM with look-ahead (cheating) | 0.96 | 13.1% | −24.7% | 94% |

- On the S&P 500 the filter kept the return, cut volatility from 18.7% to 14.5% and **halved the worst drawdown**.
- **Look-ahead bias is large:** using hindsight probabilities lifts Sharpe from 0.67 to 0.96.
- **It does not help everywhere.** For gold (0.57 vs 0.63), WTI crude (0.27 vs 0.33) and 10-year Treasuries (−0.04 vs
  −0.03) the filter did not beat buy & hold. High volatility in those markets is not reliably bad news for a long
  position (gold often rallies in stress), so this is an equity risk tool.
- The model found 3 regimes for the S&P 500: calm (8% annual volatility), normal (19%) and stressed (49%).
- **Crisis detection (S&P 500):** the 2008 crisis was flagged from day one (96% of days), COVID within 7 days (88%).
  The **2022 rate shock was mostly missed** (12% of days): prices fell steadily without extreme volatility, which a
  volatility-based regime model is not built to catch.
- **SpreadTrader by regime:** the spread portfolio's Sharpe was 1.38 in stressed S&P 500 regimes vs 0.57 in calm ones,
  but stressed days are only 3.6% of the sample, so treat it as a hint, not a finding.

*Numbers are from the run on 19 Nov 2026 and move slightly with each weekly update.*

## How it works

```
futures prices ──► daily log returns
        │
        ├─ every 6 months: Baum-Welch EM on all data so far (K = 2 and 3, several starts) ──► pick K by BIC
        │                                                   │
        │                                  forward filter: P(regime_t | returns up to t)
        │                                                   │
        └─ at today's close: long if P(stressed) < 0.5 else flat ──► tomorrow's return − 2 bp per switch
                                                            │
             compare with buy & hold, a simple volatility rule, and smoothed (look-ahead) probabilities
```

| File | Purpose |
|---|---|
| `regimelab/hmm.py` | Gaussian HMM from scratch: emissions, scaled forward-backward, Baum-Welch, BIC, simulation |
| `regimelab/regime.py` | Walk-forward refits, filtered probabilities, full-sample fit, per-regime statistics |
| `regimelab/strategy.py` | Positions, costs, volatility-rule baseline |
| `regimelab/run.py` | Whole study, crisis check, SpreadTrader link, dashboard data |
| `site/index.html` | Dashboard |
| `GUIDE.md` | Plain-language explanation of every concept, with interview questions |

## Run it yourself

```bash
pip install -r requirements.txt yfinance pytest
python -m regimelab.data      # downloads data/prices.csv
python -m regimelab.run       # writes site/data and results/RESULTS.md
pytest -q                     # 16 tests
python -m http.server -d site 8000
```

## Tests

16 tests, mostly on data simulated from a known HMM: the fit recovers the true volatilities and transition
probabilities, BIC picks the true number of regimes, **EM never lowers the likelihood**, probabilities sum to one,
**the filter uses no future data while the smoother does**, the walk-forward output before a date does not change when
later data is altered, positions earn the next day's return, costs are charged per switch, and WTI's negative price
is removed.

## Limitations

- Yahoo Finance continuous front-month prices jump on roll days.
- Daily closes; one position size; no leverage or short positions.
- Normal returns within each regime understate fat tails; the stressed regime partly absorbs them.
- The regime is defined by volatility, so slow, low-volatility declines (2022) are hard to detect.
- Research, not trading advice.

## License

MIT
