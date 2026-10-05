"""
Real NIFTY option prices (NSE F&O bhavcopy, monthly expiries) -> implied
forward, at-the-money implied volatility and a 30-day constant-maturity
ATM IV series that can be compared directly with India VIX.

Method, per trading day and monthly expiry
------------------------------------------
forward   F = K* + e^{rT} (C(K*) - P(K*)),  K* = strike with the smallest |C - P|
          among strikes where both legs traded (put-call parity; the tests
          check it against the NIFTY futures price)
ATM IV    Black-76 implied vol of the call and put at the traded strike
          closest to F, averaged
IV30      total implied variance (sigma^2 * T) interpolated linearly in T
          between the two expiries that bracket 30 calendar days
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

DATA = Path(__file__).resolve().parents[1] / "data"
R = 0.065


def black76(F, K, T, sig, r=R, kind="C"):
    sq = sig * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sig**2 * T) / sq
    d2 = d1 - sq
    df = np.exp(-r * T)
    if kind == "C":
        return df * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return df * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def black76_delta(F, K, T, sig, r=R, kind="C"):
    d1 = (np.log(F / K) + 0.5 * sig**2 * T) / (sig * np.sqrt(T))
    return np.exp(-r * T) * (norm.cdf(d1) if kind == "C" else norm.cdf(d1) - 1)


def implied_vol(price, F, K, T, kind, r=R):
    intrinsic = np.exp(-r * T) * max((F - K) if kind == "C" else (K - F), 0.0)
    if not np.isfinite(price) or price <= intrinsic + 1e-6 or T <= 0:
        return np.nan
    try:
        return brentq(lambda s: black76(F, K, T, s, r, kind) - price, 1e-4, 5.0, xtol=1e-8)
    except ValueError:
        return np.nan


def load_chain(path: Path = DATA / "nifty_monthly_options_eod.csv.gz") -> pd.DataFrame:
    o = pd.read_csv(path, parse_dates=["date", "expiry"])
    o["T"] = (o["expiry"] - o["date"]).dt.days / 365.0
    return o


def expiry_snapshot(g: pd.DataFrame) -> dict | None:
    """Forward and ATM IV for one (date, expiry) slice of the chain."""
    w = g.pivot_table(index="strike", columns="type", values=["close", "contracts"], aggfunc="last")
    if ("close", "CE") not in w or ("close", "PE") not in w:
        return None
    traded = w[(w[("contracts", "CE")] > 0) & (w[("contracts", "PE")] > 0)]
    if traded.empty:
        return None
    T = g["T"].iloc[0]
    diff = traded[("close", "CE")] - traded[("close", "PE")]
    k_star = diff.abs().idxmin()
    F = k_star + np.exp(R * T) * diff.loc[k_star]
    k_atm = traded.index[np.argmin(np.abs(traded.index - F))]
    c, p = traded.loc[k_atm, ("close", "CE")], traded.loc[k_atm, ("close", "PE")]
    ivs = [implied_vol(c, F, k_atm, T, "C"), implied_vol(p, F, k_atm, T, "P")]
    iv = np.nanmean(ivs) if np.isfinite(ivs).any() else np.nan
    return {"forward": F, "atm_strike": k_atm, "call": c, "put": p, "atm_iv": iv, "T": T}


def atm_term_structure(chain: pd.DataFrame, min_days: int = 1) -> pd.DataFrame:
    rows = []
    for (date, expiry), g in chain[(chain["expiry"] - chain["date"]).dt.days >= min_days].groupby(["date", "expiry"]):
        s = expiry_snapshot(g)
        if s:
            rows.append({"date": date, "expiry": expiry, **s})
    return pd.DataFrame(rows)


def iv30(ts: pd.DataFrame, days: int = 30) -> pd.Series:
    """30-day constant-maturity ATM IV (%) by total-variance interpolation."""
    target = days / 365
    out = {}
    for date, g in ts.dropna(subset=["atm_iv"]).groupby("date"):
        g = g.sort_values("T")
        lo, hi = g[g["T"] <= target], g[g["T"] > target]
        if len(lo) and len(hi):
            a, b = lo.iloc[-1], hi.iloc[0]
            w = (target - a["T"]) / (b["T"] - a["T"])
            var = (1 - w) * a["atm_iv"] ** 2 * a["T"] + w * b["atm_iv"] ** 2 * b["T"]
            out[date] = np.sqrt(var / target) * 100
        elif len(hi):
            out[date] = hi.iloc[0]["atm_iv"] * 100
    return pd.Series(out, name="atm_iv30")


def build_iv30(cache: Path = DATA / "nifty_atm_iv.csv") -> tuple[pd.DataFrame, pd.Series]:
    """Term structure and IV30, cached to data/ because the inversion takes ~1 min."""
    if cache.exists():
        ts = pd.read_csv(cache, parse_dates=["date", "expiry"])
    else:
        ts = atm_term_structure(load_chain())
        ts.to_csv(cache, index=False, date_format="%Y-%m-%d")
    return ts, iv30(ts)


if __name__ == "__main__":
    from data import load
    ts, s = build_iv30()
    df = load()
    comp = pd.DataFrame({"vix": df["vix"], "atm_iv30": s}).dropna()
    diff = comp.vix - comp.atm_iv30
    print(f"{len(comp)} days  {comp.index[0].date()} to {comp.index[-1].date()}")
    print(f"mean VIX {comp.vix.mean():.2f}, mean ATM IV30 {comp.atm_iv30.mean():.2f}")
    print(f"VIX - ATM IV30: mean {diff.mean():.2f}, median {diff.median():.2f}, "
          f"p10 {diff.quantile(.1):.2f}, p90 {diff.quantile(.9):.2f}, corr {comp.corr().iloc[0, 1]:.3f}")
    fut = pd.read_csv(DATA / "nifty_futures_eod.csv", parse_dates=["date", "expiry"])
    m = ts.merge(fut, on=["date", "expiry"])
    print(f"parity forward vs futures: median |diff| {(m.forward - m.settle).abs().median():.2f} pts "
          f"({((m.forward - m.settle).abs() / m.settle).median() * 1e4:.1f} bps), n={len(m)}")
