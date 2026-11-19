import numpy as np
import pandas as pd
import pytest

from regimelab.data import log_returns_pct
from regimelab.hmm import (HMMParams, filter_probs, fit, fit_once, loglik, select_k,
                           simulate, smooth_probs, _initial)
from regimelab.metrics import max_drawdown, summarize
from regimelab.regime import walk_forward
from regimelab.strategy import apply_positions, threshold_position, vol_rule_position

TRUE = HMMParams(np.array([0.5, 0.5]), np.array([[0.98, 0.02], [0.05, 0.95]]),
                 np.array([0.05, -0.10]), np.array([0.6 ** 2, 2.0 ** 2]))


@pytest.fixture(scope="module")
def sample():
    return simulate(TRUE, 4000, seed=1)


# ---- HMM ----------------------------------------------------------------------------

def test_fit_recovers_known_parameters(sample):
    x, _ = sample
    p = fit(x, 2, restarts=3)
    assert np.sqrt(p.var) == pytest.approx([0.6, 2.0], rel=0.1)
    assert np.diag(p.A) == pytest.approx([0.98, 0.95], abs=0.02)


def test_states_are_ordered_calm_to_stressed(sample):
    x, _ = sample
    p = fit(x, 3, restarts=3)
    assert np.all(np.diff(p.var) > 0)


def test_bic_picks_two_states_for_two_state_data(sample):
    x, _ = sample
    best, info = select_k(x, (1, 2, 3), restarts=3)
    assert best.k == 2
    assert info["bic"][2] < info["bic"][1]


def test_em_never_decreases_likelihood(sample):
    x, _ = sample
    rng = np.random.default_rng(0)
    p = _initial(x, 2, rng, jitter=True)
    lls = [loglik(x, p)]
    for _ in range(15):
        p = fit_once(x, p, max_iter=1)
        lls.append(p.loglik)
    assert all(b >= a - 1e-6 for a, b in zip(lls, lls[1:]))


def test_probabilities_sum_to_one(sample):
    x, _ = sample
    f, s = filter_probs(x, TRUE), smooth_probs(x, TRUE)
    np.testing.assert_allclose(f.sum(axis=1), 1.0, atol=1e-9)
    np.testing.assert_allclose(s.sum(axis=1), 1.0, atol=1e-9)


def test_filter_uses_no_future_data(sample):
    x, _ = sample
    full = filter_probs(x, TRUE)
    part = filter_probs(x[:1500], TRUE)
    np.testing.assert_allclose(full[:1500], part, rtol=1e-12)


def test_smoother_does_use_future_data(sample):
    x, _ = sample
    full = smooth_probs(x, TRUE)
    part = smooth_probs(x[:1500], TRUE)
    assert np.max(np.abs(full[1400:1500] - part[1400:1500])) > 1e-3


def test_filter_identifies_true_states(sample):
    x, s = sample
    acc_f = np.mean(filter_probs(x, TRUE).argmax(1) == s)
    acc_s = np.mean(smooth_probs(x, TRUE).argmax(1) == s)
    assert acc_f > 0.85
    assert acc_s >= acc_f                       # hindsight helps, which is the point


def test_expected_duration():
    assert TRUE.expected_duration() == pytest.approx([50.0, 20.0])


# ---- walk-forward --------------------------------------------------------------------

def test_walk_forward_has_no_look_ahead():
    x, _ = simulate(TRUE, 1800, seed=4)
    r = pd.Series(x, pd.bdate_range("2010-01-01", periods=len(x)))
    a = walk_forward(r, ks=(2,), min_train=600, refit_every=200, restarts=2).probs
    r2 = r.copy()
    r2.iloc[1200:] = r2.iloc[1200:] * 5          # wreck the future
    b = walk_forward(r2, ks=(2,), min_train=600, refit_every=200, restarts=2).probs
    pd.testing.assert_series_equal(a["p_stressed"].iloc[:1200], b["p_stressed"].iloc[:1200])
    assert a["p_stressed"].iloc[:600].isna().all()     # nothing before the first fit


# ---- strategy ------------------------------------------------------------------------

def test_positions_apply_to_next_day():
    r = pd.Series([0.01, 0.02, -0.03, 0.04], pd.bdate_range("2020-01-01", periods=4))
    pos = pd.Series([1.0, 0.0, 0.0, 1.0], r.index)
    out = apply_positions(r, pos, cost_bps=0)
    assert out.tolist() == [0.0, 0.02, 0.0, 0.0]


def test_costs_charged_on_changes():
    r = pd.Series([0.0] * 4, pd.bdate_range("2020-01-01", periods=4))
    pos = pd.Series([1.0, 0.0, 1.0, 1.0], r.index)
    out = apply_positions(r, pos, cost_bps=10)
    assert out.sum() == pytest.approx(-3 * 0.001)


def test_threshold_and_vol_rule():
    p = pd.Series([0.1, 0.6, np.nan])
    assert threshold_position(p).iloc[:2].tolist() == [1.0, 0.0]
    assert np.isnan(threshold_position(p).iloc[2])
    rng = np.random.default_rng(0)
    r = pd.Series(np.r_[rng.normal(0, 1, 400), rng.normal(0, 4, 100)])
    pos = vol_rule_position(r)
    assert pos.iloc[-50:].mean() < 0.2              # flat in the high-volatility tail


def test_log_returns_drop_negative_prices():
    p = pd.Series([18.3, -37.6, 10.0], pd.bdate_range("2020-04-17", periods=3))
    r = log_returns_pct(p)
    assert len(r) == 1 and r.iloc[0] == pytest.approx(100 * np.log(10.0 / 18.3))


def test_metrics():
    assert max_drawdown(pd.Series([0.1, -0.5, 0.2])) == pytest.approx(-0.5)
    s = summarize(pd.Series([0.001] * 252), pd.Series([1.0] * 252))
    assert s["time_invested"] == 1.0 and s["max_drawdown"] == 0.0


def test_full_asset_analysis_runs_end_to_end():
    from regimelab.run import analyse_asset
    x, _ = simulate(TRUE, 1300, seed=9)
    prices = pd.Series(100 * np.exp(np.cumsum(x / 100)), pd.bdate_range("2015-01-01", periods=1300), name="ES=F")
    res = analyse_asset(prices, "test")
    m = res["summary"]["metrics"]
    assert set(m) == {"buy_and_hold", "hmm_filter", "vol_rule", "hmm_lookahead"}
    d = res["detail"]
    assert len(d["dates"]) == len(d["p_stressed"]) == len(d["label"])
