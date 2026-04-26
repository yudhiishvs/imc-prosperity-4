"""
Black-Scholes utilities for IMC Prosperity 4 options analysis.

All volatilities are expressed per Solvenarian day (the competition's time unit).
TTE (time-to-expiry) is measured in Solvenarian days.
"""

import numpy as np
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Core BS functions (vectorized over numpy arrays)
# ---------------------------------------------------------------------------

def _broadcast_K(K, n):
    """Return K as a numpy array of length n (handles scalar or array K)."""
    K = np.asarray(K, dtype=float)
    if K.ndim == 0:
        return np.full(n, float(K))
    return K


def bs_call(S, K, T, sigma, r=0.0):
    """
    Vectorized Black-Scholes European call price.

    Parameters
    ----------
    S : array-like  spot price(s)
    K : float or array-like  strike price(s)
    T : array-like  time to expiry (Solvenarian days)
    sigma : array-like  volatility (per √Solvenarian-day)
    r : float       risk-free rate per day (default 0)

    Returns
    -------
    numpy array of call prices
    """
    S = np.asarray(S, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    K_arr = _broadcast_K(K, len(S))

    valid = (T > 1e-10) & (sigma > 1e-10) & (S > 0)
    price = np.where(S > K_arr, S - K_arr, 0.0).astype(float)

    if valid.any():
        Sv, Kv, Tv, sv = S[valid], K_arr[valid], T[valid], sigma[valid]
        sqrt_T = np.sqrt(Tv)
        d1 = (np.log(Sv / Kv) + (r + 0.5 * sv**2) * Tv) / (sv * sqrt_T)
        d2 = d1 - sv * sqrt_T
        price[valid] = Sv * norm.cdf(d1) - Kv * np.exp(-r * Tv) * norm.cdf(d2)

    return price


def bs_vega(S, K, T, sigma, r=0.0):
    """Vectorized BS vega (dC/dσ)."""
    S = np.asarray(S, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    K_arr = _broadcast_K(K, len(S))

    valid = (T > 1e-10) & (sigma > 1e-10) & (S > 0)
    vega = np.zeros_like(S, dtype=float)

    if valid.any():
        Sv, Kv, Tv, sv = S[valid], K_arr[valid], T[valid], sigma[valid]
        sqrt_T = np.sqrt(Tv)
        d1 = (np.log(Sv / Kv) + (r + 0.5 * sv**2) * Tv) / (sv * sqrt_T)
        vega[valid] = Sv * norm.pdf(d1) * sqrt_T

    return vega


def implied_vol(C_market, S, K, T, r=0.0, init_sigma=0.5, max_iter=60, tol=1e-5):
    """
    Vectorized implied volatility via Newton-Raphson.

    Parameters
    ----------
    C_market : array-like  observed call mid-prices
    S        : array-like  spot prices (same length)
    K        : float or array-like  strike(s)
    T        : array-like  time to expiry (same length)

    Returns
    -------
    numpy array of IVs; np.nan where not solvable
    """
    C_market = np.asarray(C_market, dtype=float)
    S = np.asarray(S, dtype=float)
    T = np.asarray(T, dtype=float)
    K_arr = _broadcast_K(K, len(S))

    # Brenner-Subrahmanyam approximation as initial guess (better than a fixed value)
    T_safe = np.maximum(T, 1e-10)
    bs_approx = np.sqrt(2 * np.pi / T_safe) * C_market / S
    sigma = np.clip(bs_approx, 1e-5, 5.0)
    sigma = np.where(np.isnan(sigma) | (sigma <= 0), init_sigma, sigma)

    for _ in range(max_iter):
        C = bs_call(S, K_arr, T, sigma, r)
        V = bs_vega(S, K_arr, T, sigma, r)

        err = C - C_market
        active = (V > 1e-10) & (T > 1e-10)
        # Dampened Newton step to prevent overshooting
        step = np.where(active, err / (V + 1e-20), 0.0)
        step = np.clip(step, -sigma * 0.5, sigma * 0.5)  # max 50% change per step
        sigma -= step
        sigma = np.clip(sigma, 1e-6, 20.0)

        if active.any() and np.max(np.abs(err[active])) < tol:
            break

    # Invalidate nonsensical results
    intrinsic = np.maximum(S - K_arr, 0.0)
    bad = (C_market <= intrinsic + 1e-6) | (T <= 0) | (S <= 0) | np.isnan(C_market)
    sigma[bad] = np.nan

    return sigma


# ---------------------------------------------------------------------------
# Smile fitting
# ---------------------------------------------------------------------------

def log_moneyness(S, K, T):
    """
    Standardised log-moneyness: log(S/K) / sqrt(T).
    Positive = in-the-money (S > K), negative = out-of-the-money.
    Returns np.nan where T <= 0 or S <= 0.
    """
    S = np.asarray(S, dtype=float)
    T = np.asarray(T, dtype=float)
    valid = (T > 1e-10) & (S > 0)
    m = np.full_like(S, np.nan)
    m[valid] = np.log(S[valid] / K) / np.sqrt(T[valid])
    return m


def fit_smile_parabola(moneyness_arr, iv_arr):
    """
    Fit quadratic parabola a·m² + b·m + c to (moneyness, IV) data.

    Returns
    -------
    coeffs : ndarray shape (3,) or None if < 3 valid points
    """
    m = np.asarray(moneyness_arr, dtype=float)
    iv = np.asarray(iv_arr, dtype=float)
    valid = ~(np.isnan(m) | np.isnan(iv))
    if valid.sum() < 3:
        return None
    return np.polyfit(m[valid], iv[valid], 2)


def smile_iv_from_coeffs(moneyness_val, coeffs):
    """Evaluate fitted parabola at given moneyness value(s)."""
    if coeffs is None:
        return np.full_like(np.asarray(moneyness_val, dtype=float), np.nan)
    return np.polyval(coeffs, moneyness_val)


# ---------------------------------------------------------------------------
# Greeks
# ---------------------------------------------------------------------------

def bs_greeks(S, K, T, sigma, r=0.0):
    """
    Vectorized BS Greeks for call options.
    K can be a scalar or array-like of same length as S.

    Returns
    -------
    dict with keys 'delta', 'gamma', 'vega', 'theta' — each a numpy array
    """
    S = np.asarray(S, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    K_arr = _broadcast_K(K, len(S))

    valid = (T > 1e-10) & (sigma > 1e-10) & (S > 0) & ~np.isnan(sigma)

    delta = np.where(S > K_arr, 1.0, 0.0).astype(float)
    gamma = np.zeros_like(S, dtype=float)
    vega  = np.zeros_like(S, dtype=float)
    theta = np.zeros_like(S, dtype=float)

    if valid.any():
        Sv, Kv, Tv, sv = S[valid], K_arr[valid], T[valid], sigma[valid]
        sqrt_T = np.sqrt(Tv)
        d1 = (np.log(Sv / Kv) + (r + 0.5 * sv**2) * Tv) / (sv * sqrt_T)
        d2 = d1 - sv * sqrt_T

        delta[valid] = norm.cdf(d1)
        gamma[valid] = norm.pdf(d1) / (Sv * sv * sqrt_T)
        vega[valid]  = Sv * norm.pdf(d1) * sqrt_T
        theta[valid] = (-(Sv * norm.pdf(d1) * sv) / (2 * sqrt_T)
                        - r * Kv * np.exp(-r * Tv) * norm.cdf(d2))

    return {'delta': delta, 'gamma': gamma, 'vega': vega, 'theta': theta}
