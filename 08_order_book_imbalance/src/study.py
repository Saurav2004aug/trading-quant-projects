"""
Does order-book imbalance predict AAPL's next price move, and can a
trader make money from it after paying the spread?

Split: models are fitted on the morning (09:35-12:45) and evaluated on the
afternoon (12:45-15:55) of 21 June 2012. One day of one stock is a case
study, not a general result; everything below is out-of-sample within it.

Sections
--------
1. Contemporaneous: price change over a window vs order-flow imbalance over
   the same window (Cont, Kukanov & Stoikov 2014). Explains, doesn't predict.
2. Predictive, clock time: future mid change on current imbalance, depth
   imbalance, microprice offset and past OFI, horizons 1-60 s.
3. Predictive, event time: probability the next mid change is up, by imbalance.
4. Tradability: act on the prediction as a liquidity taker (buy at the ask,
   sell at the bid) and compare the gross signal with the cost of crossing.
5. The market maker's side: mid-price move after trades (adverse selection).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from features import clock_grid, event_features, next_mid_move
from lobster import TICK, load

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)
HORIZONS = (1, 5, 10, 30, 60)
FEATURES = ["imb1", "imb3", "micro_minus_mid_ticks", "ofi_past"]
TAKER_FEE_TICKS = 0.3            # ~$0.003/share NASDAQ taker fee = 0.3 ticks per side


def ols_fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    X1 = np.column_stack([np.ones(len(X)), X])
    return np.linalg.lstsq(X1, y, rcond=None)[0]


def ols_predict(beta, X):
    return np.column_stack([np.ones(len(X)), X]) @ beta


def r2(y, yhat):
    return 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)


def split(g: pd.DataFrame):
    cut = g["time"].median()
    return g[g.time < cut], g[g.time >= cut]


def contemporaneous(train, test) -> pd.DataFrame:
    rows = []
    for h in HORIZONS:
        x, y = f"ofi_next_{h}s", f"dmid_{h}s"
        beta = ols_fit(train[[x]].to_numpy() / 1000, train[y].to_numpy())
        rows.append({"horizon_s": h, "ticks_per_1000_shares": beta[1],
                     "r2_train": r2(train[y].to_numpy(), ols_predict(beta, train[[x]].to_numpy() / 1000)),
                     "r2_test": r2(test[y].to_numpy(), ols_predict(beta, test[[x]].to_numpy() / 1000))})
    return pd.DataFrame(rows)


def predictive(train, test) -> tuple[pd.DataFrame, dict]:
    rows, models = [], {}
    for h in HORIZONS:
        y = f"dmid_{h}s"
        beta = ols_fit(train[FEATURES].to_numpy(), train[y].to_numpy())
        models[h] = beta
        pred = ols_predict(beta, test[FEATURES].to_numpy())
        yt = test[y].to_numpy()
        top = np.abs(pred) >= np.quantile(np.abs(pred), 0.9)
        moved = top & (yt != 0)
        rows.append({"horizon_s": h, "r2_test": r2(yt, pred), "corr_test": np.corrcoef(pred, yt)[0, 1],
                     "hit_rate_top10pct": float((np.sign(pred[moved]) == np.sign(yt[moved])).mean()),
                     "mean_abs_pred_top10pct_ticks": float(np.abs(pred[top]).mean()),
                     "realised_move_in_pred_direction_ticks": float((np.sign(pred[top]) * yt[top]).mean())})
    return pd.DataFrame(rows), models


def event_time(nm: pd.DataFrame, cut: float) -> pd.DataFrame:
    t = nm[nm.time >= cut].copy()
    t["bucket"] = pd.cut(t.imb1, np.linspace(-1, 1, 11))
    t["tight"] = t.spread_ticks <= 5
    out = t.groupby("bucket", observed=True).agg(p_up=("next_move", lambda s: (s > 0).mean()),
                                                 n=("next_move", "size")).reset_index()
    tight = t[t.tight].groupby("bucket", observed=True)["next_move"].apply(lambda s: (s > 0).mean())
    out["p_up_spread_le_5_ticks"] = out["bucket"].map(tight).astype(float)
    out["bucket_mid"] = [b.mid for b in out["bucket"]]
    return out.drop(columns="bucket")


def taker_strategy(test: pd.DataFrame, beta, h: int, threshold: float) -> dict:
    """Enter when |prediction| > threshold, as a taker, exit h seconds later as a taker."""
    pred = ols_predict(beta, test[FEATURES].to_numpy())
    side = np.where(pred > threshold, 1, np.where(pred < -threshold, -1, 0))
    spread_now = test["spread_ticks"].to_numpy()
    j = test.index.to_numpy()
    fut_spread = test["spread_ticks"].shift(-h).to_numpy()          # grid dt = 1 s
    dmid = test[f"dmid_{h}s"].to_numpy()
    ok = (side != 0) & ~np.isnan(fut_spread)
    gross_mid = side[ok] * dmid[ok]                                 # if you could trade at mid
    cross = 0.5 * spread_now[ok] + 0.5 * fut_spread[ok]            # half-spread in + half-spread out
    net = gross_mid - cross - 2 * TAKER_FEE_TICKS
    return {"horizon_s": h, "threshold_ticks": threshold, "trades": int(ok.sum()),
            "gross_at_mid_ticks": float(gross_mid.mean()) if ok.any() else np.nan,
            "spread_cost_ticks": float(cross.mean()) if ok.any() else np.nan,
            "fees_ticks": 2 * TAKER_FEE_TICKS,
            "net_ticks": float(net.mean()) if ok.any() else np.nan,
            "share_profitable": float((net > 0).mean()) if ok.any() else np.nan}


def adverse_selection(ev: pd.DataFrame, raw: pd.DataFrame, lags=(0.1, 1, 5, 10, 30)) -> pd.DataFrame:
    """Mid move after visible executions, signed in the aggressor's direction (ticks)."""
    ex = raw.index[raw["type"] == 4].to_numpy()
    aggressor = -raw.loc[ex, "direction"].to_numpy()                # resting sell (-1) hit by a buyer (+1)
    t, mid = ev["time"].to_numpy(), ev["mid"].to_numpy()
    before = mid[np.maximum(ex - 1, 0)]
    rows = []
    for lag in lags:
        j = np.searchsorted(t, t[ex] + lag, side="right") - 1
        move = aggressor * (mid[j] - before) / TICK
        rows.append({"seconds_after_trade": lag, "mean_move_ticks": move.mean(),
                     "half_spread_ticks": (ev["spread_ticks"].to_numpy()[np.maximum(ex - 1, 0)] / 2).mean()})
    return pd.DataFrame(rows)


def run() -> dict:
    raw = load()
    ev = event_features(raw)
    g = clock_grid(ev, horizons=HORIZONS)
    train, test = split(g)
    cont = contemporaneous(train, test)
    pred, models = predictive(train, test)
    nm = next_mid_move(ev)
    et = event_time(nm, g.time.median())
    trades = pd.DataFrame([taker_strategy(test, models[h], h, q)
                           for h in (1, 5, 10)
                           for q in (np.quantile(np.abs(ols_predict(models[h], train[FEATURES].to_numpy())), p)
                                     for p in (0.9, 0.99))])
    adv = adverse_selection(ev, raw)
    stats = {"events": len(raw), "grid_points": len(g), "median_spread_ticks": float(g.spread_ticks.median()),
             "share_spread_1_tick": float((g.spread_ticks == 1).mean()),
             "mid_changes": int((np.diff(ev["mid"].to_numpy()) != 0).sum())}
    for name, df in [("contemporaneous_ofi", cont), ("predictive", pred), ("event_time", et),
                     ("taker_strategy", trades), ("adverse_selection", adv)]:
        df.to_csv(RESULTS / f"{name}.csv", index=False)
    write_summary(stats, cont, pred, et, trades, adv)
    return {"ev": ev, "grid": g, "train": train, "test": test, "cont": cont, "pred": pred,
            "event_time": et, "trades": trades, "adverse": adv, "stats": stats, "models": models}


def write_summary(stats, cont, pred, et, trades, adv):
    L = ["# Order-book imbalance: AAPL, NASDAQ, 21 June 2012 (LOBSTER)", "",
         f"{stats['events']:,} book events, {stats['mid_changes']:,} mid-price changes, "
         f"median spread {stats['median_spread_ticks']:.0f} ticks "
         f"(one tick = $0.01; spread is exactly one tick {stats['share_spread_1_tick']:.1%} of the time).", "",
         "Fitted 09:35-12:45, evaluated 12:45-15:55.", "",
         "## 1. Contemporaneous order-flow imbalance", "", cont.to_markdown(index=False, floatfmt=".3f"), "",
         "## 2. Predicting the mid-price change (out of sample)", "", pred.to_markdown(index=False, floatfmt=".3f"), "",
         "## 3. Event time: P(next mid change is up) by level-1 imbalance (afternoon)", "",
         et.to_markdown(index=False, floatfmt=".3f"), "",
         "## 4. Acting on it as a liquidity taker (afternoon, ticks per share per trade)", "",
         trades.to_markdown(index=False, floatfmt=".2f"), "",
         "## 5. Mid move after trades, in the aggressor's direction", "",
         adv.to_markdown(index=False, floatfmt=".2f")]
    (RESULTS / "summary.md").write_text("\n".join(L))


if __name__ == "__main__":
    run()
    print((RESULTS / "summary.md").read_text())
