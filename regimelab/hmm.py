"""Gaussian Hidden Markov Model, written from scratch.

The market is assumed to be in one of K hidden states ("regimes", e.g. calm
and stressed). Each day:

    state_t   follows a Markov chain:  P(state_t = j | state_{t-1} = i) = A[i, j]
    return_t  | state_t = k  ~  Normal(mu_k, var_k)

Three things we need from the model:

* fit()     - learn pi, A, mu, var from returns with the EM algorithm
              (Baum-Welch), several starting points, keep the best.
* filter()  - P(state_t | returns up to and including day t). Uses no future
              data, so it is the only one allowed in a trading rule.
* smooth()  - P(state_t | ALL returns, including later days). Better for
              describing history, but it looks into the future, so a strategy
              built on it is cheating. The project shows how much.

All inner loops are compiled with numba when it is installed; otherwise they
run as plain Python (same results, slower).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:  # pragma: no cover - import guard
    from numba import njit
except ImportError:  # pragma: no cover
    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f

LOG2PI = np.log(2 * np.pi)


@njit(cache=True)
def _emission(x, mu, var):
    """Returns (B, m): B[t, k] = density / exp(m[t]), scaled so the largest is 1."""
    n, k = x.shape[0], mu.shape[0]
    logb = np.empty((n, k))
    for t in range(n):
        for j in range(k):
            d = x[t] - mu[j]
            logb[t, j] = -0.5 * (LOG2PI + np.log(var[j]) + d * d / var[j])
    m = np.empty(n)
    B = np.empty((n, k))
    for t in range(n):
        mx = logb[t, 0]
        for j in range(1, k):
            if logb[t, j] > mx:
                mx = logb[t, j]
        m[t] = mx
        for j in range(k):
            B[t, j] = np.exp(logb[t, j] - mx)
    return B, m


@njit(cache=True)
def _forward(pi, A, B):
    """Scaled forward pass. alpha[t] = P(state_t | x_1..t). Returns alpha, c."""
    n, k = B.shape
    alpha = np.empty((n, k))
    c = np.empty(n)
    s = 0.0
    for j in range(k):
        alpha[0, j] = pi[j] * B[0, j]
        s += alpha[0, j]
    c[0] = s
    for j in range(k):
        alpha[0, j] /= s
    for t in range(1, n):
        s = 0.0
        for j in range(k):
            acc = 0.0
            for i in range(k):
                acc += alpha[t - 1, i] * A[i, j]
            alpha[t, j] = acc * B[t, j]
            s += alpha[t, j]
        c[t] = s
        for j in range(k):
            alpha[t, j] /= s
    return alpha, c


@njit(cache=True)
def _backward(A, B, c):
    n, k = B.shape
    beta = np.empty((n, k))
    for j in range(k):
        beta[n - 1, j] = 1.0
    for t in range(n - 2, -1, -1):
        for i in range(k):
            acc = 0.0
            for j in range(k):
                acc += A[i, j] * B[t + 1, j] * beta[t + 1, j]
            beta[t, i] = acc / c[t + 1]
    return beta


@njit(cache=True)
def _xi_sum(alpha, beta, A, B, c):
    """Expected number of i -> j transitions."""
    n, k = B.shape
    xs = np.zeros((k, k))
    for t in range(n - 1):
        for i in range(k):
            for j in range(k):
                xs[i, j] += alpha[t, i] * A[i, j] * B[t + 1, j] * beta[t + 1, j] / c[t + 1]
    return xs


@dataclass
class HMMParams:
    pi: np.ndarray
    A: np.ndarray
    mu: np.ndarray
    var: np.ndarray
    loglik: float = np.nan
    n_iter: int = 0

    @property
    def k(self) -> int:
        return len(self.mu)

    def n_params(self) -> int:
        k = self.k
        return (k - 1) + k * (k - 1) + 2 * k

    def bic(self, n_obs: int) -> float:
        """Bayesian Information Criterion: fit penalised by number of parameters. Lower is better."""
        return -2.0 * self.loglik + self.n_params() * np.log(n_obs)

    def expected_duration(self) -> np.ndarray:
        """Average number of days a regime lasts once entered: 1 / (1 - A[k, k])."""
        return 1.0 / np.maximum(1.0 - np.diag(self.A), 1e-12)

    def to_dict(self) -> dict:
        return {"pi": self.pi.tolist(), "A": self.A.tolist(), "mu": self.mu.tolist(),
                "var": self.var.tolist(), "loglik": float(self.loglik), "n_iter": int(self.n_iter)}


def loglik(x: np.ndarray, p: HMMParams) -> float:
    B, m = _emission(np.asarray(x, float), p.mu, p.var)
    _, c = _forward(p.pi, p.A, B)
    return float(np.sum(np.log(c)) + np.sum(m))


def filter_probs(x: np.ndarray, p: HMMParams) -> np.ndarray:
    """P(state_t | x_1..x_t) for every t. No look-ahead."""
    B, _ = _emission(np.asarray(x, float), p.mu, p.var)
    alpha, _ = _forward(p.pi, p.A, B)
    return alpha


def smooth_probs(x: np.ndarray, p: HMMParams) -> np.ndarray:
    """P(state_t | x_1..x_n): uses the whole sample, including the future."""
    B, _ = _emission(np.asarray(x, float), p.mu, p.var)
    alpha, c = _forward(p.pi, p.A, B)
    beta = _backward(p.A, B, c)
    g = alpha * beta
    return g / g.sum(axis=1, keepdims=True)


def _sort_states(p: HMMParams) -> HMMParams:
    """Order states from lowest to highest volatility so labels mean something
    (EM itself returns states in arbitrary order)."""
    o = np.argsort(p.var)
    return HMMParams(p.pi[o], p.A[np.ix_(o, o)], p.mu[o], p.var[o], p.loglik, p.n_iter)


def _initial(x: np.ndarray, k: int, rng: np.random.Generator, jitter: bool) -> HMMParams:
    v = np.var(x)
    scales = np.geomspace(0.3, 3.0, k) if k > 1 else np.array([1.0])
    if jitter:
        scales = scales * np.exp(rng.normal(0, 0.3, k))
    mu = np.mean(x) + (rng.normal(0, 0.1, k) * np.sqrt(v) if jitter else np.zeros(k))
    A = np.full((k, k), 0.05 / max(k - 1, 1))
    np.fill_diagonal(A, 0.95 if k > 1 else 1.0)
    return HMMParams(np.full(k, 1.0 / k), A, mu.astype(float), (v * scales).astype(float))


def fit_once(x: np.ndarray, init: HMMParams, max_iter: int = 500, tol: float = 1e-7) -> HMMParams:
    """Baum-Welch (EM) from one starting point."""
    x = np.asarray(x, float)
    n = len(x)
    var_floor = 1e-4 * np.var(x)
    pi, A, mu, var = init.pi.copy(), init.A.copy(), init.mu.copy(), init.var.copy()
    prev = -np.inf
    ll = -np.inf
    it = 0
    for it in range(1, max_iter + 1):
        # E-step: how likely is each state on each day, given current parameters?
        B, m = _emission(x, mu, var)
        alpha, c = _forward(pi, A, B)
        beta = _backward(A, B, c)
        ll = float(np.sum(np.log(c)) + np.sum(m))
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True)
        xs = _xi_sum(alpha, beta, A, B, c)
        # M-step: re-estimate parameters as weighted averages.
        pi = gamma[0]
        A = xs / xs.sum(axis=1, keepdims=True)
        w = gamma.sum(axis=0)
        mu = (gamma * x[:, None]).sum(axis=0) / w
        var = (gamma * (x[:, None] - mu) ** 2).sum(axis=0) / w
        var = np.maximum(var, var_floor)
        if abs(ll - prev) < tol * max(1.0, abs(ll)):
            break
        prev = ll
    # Log-likelihood of the final parameters.
    final = HMMParams(pi, A, mu, var, n_iter=it)
    final.loglik = loglik(x, final)
    return final


def fit(x: np.ndarray, k: int, restarts: int = 5, seed: int = 0, max_iter: int = 500,
        warm_start: HMMParams | None = None) -> HMMParams:
    """Fit K states from several starting points and keep the best likelihood.
    EM only finds a local optimum, which is why restarts matter."""
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    inits = [_initial(x, k, rng, jitter=False)]
    inits += [_initial(x, k, rng, jitter=True) for _ in range(max(restarts - 1, 0))]
    if warm_start is not None and warm_start.k == k:
        inits.append(warm_start)
    best = None
    for init in inits:
        p = fit_once(x, init, max_iter=max_iter)
        if np.isfinite(p.loglik) and (best is None or p.loglik > best.loglik):
            best = p
    return _sort_states(best)


def select_k(x: np.ndarray, ks=(2, 3), **kw) -> tuple[HMMParams, dict]:
    """Fit each K and pick the lowest BIC."""
    fits = {k: fit(x, k, **{kk: v for kk, v in kw.items() if kk != "warm"},
                   warm_start=(kw.get("warm") or {}).get(k)) for k in ks}
    bics = {k: p.bic(len(x)) for k, p in fits.items()}
    best_k = min(bics, key=bics.get)
    return fits[best_k], {"bic": bics, "fits": fits}


def simulate(p: HMMParams, n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Draw returns and hidden states from a known model (used in tests)."""
    rng = np.random.default_rng(seed)
    s = np.empty(n, dtype=int)
    x = np.empty(n)
    s[0] = rng.choice(p.k, p=p.pi)
    for t in range(n):
        if t:
            s[t] = rng.choice(p.k, p=p.A[s[t - 1]])
        x[t] = rng.normal(p.mu[s[t]], np.sqrt(p.var[s[t]]))
    return x, s
