# RegimeLab results

Generated 2026-11-19 15:10 UTC.

## S&P 500 E-mini (ES=F), out-of-sample from 2003-09-11

| Strategy | CAGR | Volatility | Sharpe | Max drawdown | Time invested |
|---|---|---|---|---|---|
| Buy & hold | 8.9% | 18.7% | 0.55 | -57.1% | 100.0% |
| HMM regime filter (honest) | 9.0% | 14.5% | 0.67 | -27.3% | 93.9% |
| Simple volatility rule | 3.4% | 9.0% | 0.41 | -21.7% | 53.1% |
| HMM with look-ahead (cheating) | 13.1% | 13.8% | 0.96 | -24.7% | 94.2% |

Full-sample model: 3 regimes (calm, normal, stressed); annual vol calm 8%, normal 19%, stressed 49%; average duration calm 31 days, normal 24 days, stressed 20 days

## US 10-year Treasury note (ZN=F), out-of-sample from 2003-09-29

| Strategy | CAGR | Volatility | Sharpe | Max drawdown | Time invested |
|---|---|---|---|---|---|
| Buy & hold | -0.3% | 5.9% | -0.03 | -25.7% | 100.0% |
| HMM regime filter (honest) | -0.3% | 4.9% | -0.04 | -24.8% | 83.8% |
| Simple volatility rule | -0.4% | 4.0% | -0.07 | -18.0% | 53.9% |
| HMM with look-ahead (cheating) | 0.5% | 4.1% | 0.14 | -16.5% | 78.7% |

Full-sample model: 3 regimes (calm, normal, stressed); annual vol calm 4%, normal 5%, stressed 9%; average duration calm 109 days, normal 50 days, stressed 38 days

## Gold (GC=F), out-of-sample from 2003-09-15

| Strategy | CAGR | Volatility | Sharpe | Max drawdown | Time invested |
|---|---|---|---|---|---|
| Buy & hold | 10.4% | 18.3% | 0.63 | -44.4% | 100.0% |
| HMM regime filter (honest) | 7.8% | 15.4% | 0.57 | -43.8% | 85.7% |
| Simple volatility rule | 4.3% | 10.8% | 0.44 | -33.9% | 50.7% |
| HMM with look-ahead (cheating) | 12.8% | 14.6% | 0.90 | -28.1% | 90.7% |

Full-sample model: 3 regimes (calm, normal, stressed); annual vol calm 11%, normal 17%, stressed 36%; average duration calm 60 days, normal 41 days, stressed 16 days

## WTI crude oil (CL=F), out-of-sample from 2003-09-08

| Strategy | CAGR | Volatility | Sharpe | Max drawdown | Time invested |
|---|---|---|---|---|---|
| Buy & hold | 4.7% | 43.0% | 0.33 | -93.1% | 100.0% |
| HMM regime filter (honest) | 3.5% | 32.3% | 0.27 | -79.3% | 91.1% |
| Simple volatility rule | 4.9% | 22.1% | 0.33 | -76.2% | 53.1% |
| HMM with look-ahead (cheating) | 5.4% | 32.9% | 0.32 | -80.6% | 95.6% |

Full-sample model: 3 regimes (calm, normal, stressed); annual vol calm 25%, normal 43%, stressed 133%; average duration calm 55 days, normal 32 days, stressed 16 days

## Did the honest model flag known stress episodes? (S&P 500)

| Episode | Avg P(stressed) | Days flagged | Days until first flag | S&P 500 return |
|---|---|---|---|---|
| Global financial crisis | 0.88 | 96.4% | 0 | -36.8% |
| US downgrade / euro crisis | 0.57 | 56.9% | 3 | -3.0% |
| China devaluation selloff | 0.37 | 38.7% | 6 | -9.1% |
| Volmageddon | 0.42 | 42.1% | 4 | -3.9% |
| COVID crash | 0.86 | 88.0% | 7 | -14.3% |
| 2022 rate shock | 0.14 | 12.0% | 116 | -18.4% |

## SpreadTrader portfolio by S&P 500 regime (from 2012-01-03)

| Regime | Share of days | Spread engine annual return | Spread engine Sharpe | S&P 500 Sharpe |
|---|---|---|---|---|
| calm | 54.8% | 0.2% | 0.57 | 0.36 |
| normal | 41.6% | 0.1% | 0.41 | 1.21 |
| stressed | 3.6% | 0.5% | 1.38 | 1.29 |
