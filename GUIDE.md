# RegimeLab: the guide to explaining this project in an interview

Read once top to bottom, then practise the questions at the end out loud.

---

## 1. The one-minute pitch

> "Markets have moods: long calm stretches and short, violent stressed ones. I
> wrote a Hidden Markov Model from scratch that estimates which mood a futures
> market is in, using only data available at the time. I fitted it walk-forward
> on 26 years of S&P 500, Treasury, gold and crude futures and used it as a risk
> switch: stay long when calm, step out when stress is likely.
>
> On the S&P 500 it kept the same return as buy-and-hold but halved the worst
> drawdown, from −57% to −27%, and lifted the Sharpe ratio from 0.55 to 0.67.
> I also showed that if you accidentally use hindsight probabilities, the Sharpe
> jumps to 0.96, which is the trap many online backtests fall into. And it did
> not help on gold or oil, which taught me it is an equity risk tool."

## 2. The idea in plain words

You cannot see the market's "mood" directly; you only see returns. But calm
days and stressed days produce different-looking returns: small and steady vs
large and jumpy. An HMM works backwards from the returns to the most likely
hidden mood.

Two assumptions:

1. **The mood follows a Markov chain.** Tomorrow's mood depends only on today's.
   Moods are "sticky": calm tends to stay calm. The **transition matrix** A holds
   these probabilities, e.g. P(calm tomorrow | calm today) = 97%.
2. **Returns depend on the mood.** In each regime k, daily returns are normal
   with their own mean and volatility.

**Expected duration** of a regime = 1 / (1 − probability of staying). If calm
stays calm with probability 0.97, a calm spell lasts about 33 days on average.

## 3. How the model is fitted: Baum-Welch (EM)

We do not know the moods, so we cannot just average returns per mood. EM goes
round in a loop:

- **E-step:** with the current guesses, compute for each day the probability of
  each mood (forward-backward algorithm).
- **M-step:** re-estimate means, volatilities and transition probabilities as
  averages weighted by those probabilities.

Each loop never makes the fit worse (a test checks this). It stops when the
likelihood stops improving. EM can get stuck in a **local optimum**, so the code
starts from several different guesses and keeps the best.

**Forward algorithm:** walks through the days once, carrying "probability of each
mood given everything up to today". Numbers get tiny over thousands of days, so
each step is rescaled (that is the "scaled" forward pass).

**Labelling:** EM returns the regimes in random order. The code sorts them by
volatility, so "calm" always means the lowest-volatility regime.

## 4. How many regimes? BIC

More regimes always fit the past better, so we need a penalty for complexity.
**BIC = −2 × log-likelihood + (number of parameters) × ln(number of days).**
Lower is better. Each refit compares 2 and 3 regimes and keeps the lower BIC.
For the S&P 500 the full history picks 3: calm (8% annual volatility), normal
(19%), stressed (49%).

## 5. Filtered vs smoothed: the trap

- **Filtered:** P(mood today | returns up to today). This is what you would have
  known on the day. Only this may be used for trading.
- **Smoothed:** P(mood today | ALL returns, including the future). Better for
  describing history (it knows a crash came next), but trading on it is
  cheating.

The project runs both. On the S&P 500 the honest version has Sharpe 0.67 and
the cheating version 0.96. **That 0.29 gap is pure look-ahead bias.** Tests
prove the filter output does not change when future data changes, and the
smoother's does.

A second, quieter source of look-ahead: fitting the model on the whole history,
even if you use filtered probabilities. That is why the trading results come
from **walk-forward** fits (refit every 6 months on past data only).

## 6. The strategy and the comparisons

At each day's close: if P(stressed) < 0.5 stay long, otherwise go flat. The
position earns the next day's return. Each switch costs 2 basis points.

Compared with:

- **Buy & hold**: the default.
- **Simple volatility rule**: flat when 20-day volatility is above its 1-year
  median. If the HMM could not beat this, it would just be a complicated way to
  measure volatility. On the S&P 500 the HMM did better (Sharpe 0.67 vs 0.41)
  because it only steps out in truly stressed periods; the simple rule is out of
  the market half the time and misses rallies.
- **Look-ahead version**: to show how much cheating flatters results.

## 7. What the results mean

- **S&P 500:** same return as buy & hold (9.0% vs 8.9% a year), lower volatility,
  worst drawdown −27% instead of −57%. Invested 94% of the time.
- **Gold, crude, Treasuries:** no improvement. In gold, stress often comes with
  rallies (safe-haven buying); in oil, volatility is high in both directions.
  So the rule "high volatility = step out" only makes sense where stress means
  falling prices, which is mainly equities.
- **Crises:** 2008 flagged from the first day; COVID within 7 days; the 2022
  rate shock mostly missed because prices fell steadily without extreme
  volatility. This shows what the model measures: volatility, not direction.
- **Link to SpreadTrader:** the spread portfolio did best when the S&P 500 was
  stressed (Sharpe 1.38 vs 0.57 in calm), but only 3.6% of days were stressed,
  so the sample is small.

## 8. Words you need

- **Sharpe ratio:** return ÷ volatility, annualised (×√252). No risk-free rate is
  subtracted because a futures position needs margin, not cash.
- **Max drawdown:** worst fall from a peak.
- **Log return:** ln(today's price / yesterday's). Adds up over days.
- **Annualised volatility:** daily standard deviation × √252.
- **Stationary / regime change:** the statistics of returns are not constant; an
  HMM is one way to model that.

## 9. Honest limitations

1. Normal distributions inside each regime understate fat tails.
2. Regimes are about volatility, not direction (2022).
3. Daily data, one position size, no shorting or leverage.
4. Continuous front-month futures data with roll jumps.
5. The threshold of 0.5 is a choice; the dashboard shows 0.3 and 0.7 too.

## 10. Questions you will get

**Why an HMM and not just a volatility filter?** It models persistence (sticky
regimes) and gives probabilities, so it switches less often on one noisy day.
And the simple rule did worse here.

**Why does the filtered probability lag?** It needs a few big-move days before it
is confident the regime has changed. COVID took 7 days to flag. That lag is the
price of not looking into the future.

**How did you pick 2 vs 3 regimes?** BIC at every refit. It penalises extra
parameters by ln(number of days).

**What is the EM algorithm guaranteed to do?** Never decrease the likelihood each
iteration; it only finds a local optimum, hence multiple starting points.

**How would you improve it?** Student-t emissions for fat tails, using more inputs
(volatility index, credit spreads, yield curve) as observations, scaling the
position by P(calm) instead of all-or-nothing, and adding short positions.

**Quick maths:** if P(stay calm) = 0.98, how long is a calm spell on average?
1 / 0.02 = 50 days.

---

## Where things are in the code

| File | What it does |
|---|---|
| `regimelab/hmm.py` | The HMM: emissions, forward, backward, Baum-Welch, BIC, simulation |
| `regimelab/regime.py` | Walk-forward refits and regime statistics |
| `regimelab/strategy.py` | Turning probabilities into positions, costs, the volatility baseline |
| `regimelab/run.py` | Runs everything, the crisis check, the SpreadTrader link |
| `site/index.html` | Dashboard |
| `tests/test_regimelab.py` | 16 tests on simulated data with known answers |
