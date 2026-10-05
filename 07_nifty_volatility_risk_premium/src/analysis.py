"""Run every analysis and write results/*.csv and results/summary.md."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data import load
from options_data import build_iv30, load_chain
from real_straddle import run_cycles, summarise
from straddle import Config, all_offsets, stats
from vrp import build, by_year, summary

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)


def exante_vrp_filter(row) -> bool:
    """Sell only when VIX is above trailing 21-day realised vol (known at entry)."""
    return bool(row["vix"] > row["rv_past"])


def attribution_r2(trades: pd.DataFrame) -> float:
    t = trades[trades.traded]
    explained = t.gamma_pct + t.theta_pct + t.vega_pct - t.cost_pct
    return 1 - (t.pnl_pct - explained).var() / t.pnl_pct.var()


def run() -> dict:
    df = build(load())
    ts, iv30 = build_iv30()
    comp = pd.DataFrame({"vix": df["vix"], "atm_iv30": iv30, "rv_fwd": df["rv_fwd"]}).dropna(subset=["vix", "atm_iv30"])
    gap = comp.vix - comp.atm_iv30
    real_vrp = (comp.atm_iv30 - comp.rv_fwd).dropna()

    chain, spot = load_chain(), df["spot"]
    real_h = run_cycles(chain, ts, spot)
    real_u = run_cycles(chain, ts, spot, hedge=False)
    real_costs = pd.DataFrame([{"opt_cost_pct_of_premium": oc * 100, "hedge_bps": hb,
                                **summarise(run_cycles(chain, ts, spot, opt_cost=oc, hedge_bps=hb))}
                               for oc, hb in [(0, 0), (0.005, 1), (0.02, 1), (0.05, 3)]])

    model = {name: stats(all_offsets(df, cfg)) for name, cfg in
             [("VIX as IV", Config()), ("VIX - 1.44 (measured gap)", Config(iv_offset=gap.mean()))]}
    model["VIX - 1.44, unhedged"] = stats(all_offsets(df, Config(iv_offset=gap.mean(), hedge=False)))
    model["VIX - 1.44, sell only if VIX > trailing RV"] = stats(
        all_offsets(df, Config(iv_offset=gap.mean()), entry_filter=exante_vrp_filter))
    haircut = pd.DataFrame([{"atm_iv_below_vix_pts": h, **stats(all_offsets(df, Config(iv_offset=h)))}
                            for h in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)])
    model_trades = all_offsets(df, Config(iv_offset=gap.mean()))

    vs = summary(df)
    out = {"vrp": vs, "gap": gap, "real_vrp": real_vrp, "comp": comp, "real_h": real_h, "real_u": real_u,
           "real_costs": real_costs, "model": pd.DataFrame(model).T, "haircut": haircut,
           "attr_r2": attribution_r2(model_trades), "df": df, "yearly": by_year(df)}

    real_h.to_csv(RESULTS / "real_straddle_cycles.csv", index=False, date_format="%Y-%m-%d")
    real_costs.to_csv(RESULTS / "real_straddle_costs.csv", index=False)
    comp.to_csv(RESULTS / "vix_vs_atm_iv30.csv", date_format="%Y-%m-%d")
    out["model"].to_csv(RESULTS / "model_strategies.csv")
    haircut.to_csv(RESULTS / "model_iv_haircut.csv", index=False)
    out["yearly"].to_csv(RESULTS / "vrp_by_year.csv")
    pd.Series(vs).to_csv(RESULTS / "vrp_summary.csv", header=["value"])
    write_summary(out)
    return out


def write_summary(o: dict) -> None:
    vs, sh, su = o["vrp"], summarise(o["real_h"]), summarise(o["real_u"])
    f = lambda x: f"{x:.2f}"  # noqa: E731
    L = ["# NIFTY volatility risk premium: results", "",
         "## 1. India VIX vs subsequent realised volatility",
         f"{vs['start']} to {vs['end']}, {vs['days']} days.", "",
         "| | |", "|---|---:|",
         f"| Mean VIX / mean realised vol (next 21 days) | {f(vs['mean_vix'])} / {f(vs['mean_rv_fwd'])} |",
         f"| Mean premium, vol points (Newey-West t) | {f(vs['mean_vrp_vol'])} ({f(vs['vrp_vol_t_nw'])}) |",
         f"| Days VIX above subsequent realised | {vs['share_vix_above_rv']:.0%} |",
         f"| Worst day | {f(vs['worst_vrp_vol'])} on {vs['worst_vrp_date']} |",
         f"| Forecast R²: VIX / trailing realised | {f(vs['mz_vix_r2'])} / {f(vs['mz_past_r2'])} |",
         f"| Forecast RMSE: VIX / trailing realised | {f(vs['rmse_vix'])} / {f(vs['rmse_past'])} |", "",
         "## 2. India VIX vs real 30-day ATM implied vol (NSE option prices)",
         f"{len(o['gap'])} days. VIX - ATM IV30: mean {f(o['gap'].mean())}, median {f(o['gap'].median())}, "
         f"10th-90th pct {f(o['gap'].quantile(.1))} to {f(o['gap'].quantile(.9))}; correlation "
         f"{o['comp'][['vix', 'atm_iv30']].corr().iloc[0, 1]:.3f}.",
         f"ATM IV30 - subsequent realised: mean {f(o['real_vrp'].mean())}, median {f(o['real_vrp'].median())}, "
         f"positive on {(o['real_vrp'] > 0).mean():.0%} of days.", "",
         "## 3. Short monthly ATM straddle at real NSE prices (per cycle, % of forward)", "",
         "| | Delta-hedged | Unhedged |", "|---|---:|---:|"]
    for k in ("cycles", "mean_pnl_pct", "median_pnl_pct", "t_stat", "win_rate", "worst_pct", "best_pct",
              "sharpe", "skew", "mean_premium_pct", "mean_iv_entry", "mean_rv_realised", "mean_cost_pct"):
        a, b = sh[k], su[k]
        L.append(f"| {k} | {a if isinstance(a, (int, np.integer)) else f(a)} | "
                 f"{b if isinstance(b, (int, np.integer)) else f(b)} |")
    L += ["", f"Period {sh['first_entry']} to {sh['last_expiry']}. Worst hedged cycle entered {sh['worst_entry']}. "
          f"Share of option marks from days a leg did not trade: {sh['stale_marks_share']:.1%}.", "",
          "Cost sensitivity (delta-hedged):", "",
          o["real_costs"][["opt_cost_pct_of_premium", "hedge_bps", "mean_pnl_pct", "t_stat", "sharpe"]]
          .to_markdown(index=False, floatfmt=".2f"), "",
          "## 4. Model on the longer 2020-2026 VIX sample (includes the March 2020 crash)", "",
          o["model"][["traded", "mean_pnl_pct", "win_rate", "worst_pct", "sharpe", "skew"]]
          .to_markdown(floatfmt=".2f"), "",
          f"Greeks attribution explains {o['attr_r2']:.0%} of the variance of model cycle P&L.", "",
          o["haircut"][["atm_iv_below_vix_pts", "mean_pnl_pct", "win_rate", "sharpe"]]
          .to_markdown(index=False, floatfmt=".2f")]
    (RESULTS / "summary.md").write_text("\n".join(L))


if __name__ == "__main__":
    run()
    print((RESULTS / "summary.md").read_text())
