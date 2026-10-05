"""
Synthetic run-to-failure fleet data (vibration RMS, temperature, load).

Designed so that the easy answer ("vibration is high -> failing") is only
partly right, as on a real shop floor:

- Machines differ: healthy vibration ranges 0.10-0.45 g and healthy
  temperature 30-55 C, so a high reading on one machine is normal on another.
- Operating load varies cycle to cycle and raises both vibration and
  temperature, so a load spike looks like degradation.
- Two failure modes:
    * gradual wear (70%): degradation starts at 50-75% of life and ramps up
    * abrupt fault (30%): little warning, degradation only in the last 8-25 cycles
- Sensor noise, slow sensor drift and ~2% missing readings.

One "cycle" = one aggregated reading (e.g. the 5-second RMS window that
the firmware publishes, or an hourly aggregate of those).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def simulate_machine(machine_id: int, life: int, rng: np.random.Generator) -> pd.DataFrame:
    cycles = np.arange(life)
    base_vib = rng.uniform(0.10, 0.45)
    base_temp = rng.uniform(30, 55)
    load = np.clip(0.7 + 0.15 * np.sin(cycles / rng.uniform(5, 15)) + rng.normal(0, 0.1, life), 0.3, 1.2)

    mode = "abrupt" if rng.random() < 0.3 else "gradual"
    if mode == "gradual":
        onset = life * rng.uniform(0.50, 0.75)
        power = rng.uniform(1.5, 2.5)
    else:
        onset = life - rng.integers(8, 26)
        power = rng.uniform(0.8, 1.2)
    deg = np.clip((cycles - onset) / (life - onset), 0, 1) ** power
    severity = rng.uniform(0.6, 1.4)

    vib = (base_vib * (0.6 + 0.5 * load) + deg * severity * rng.uniform(0.8, 1.6)
           + rng.normal(0, 0.03, life) + np.linspace(0, rng.normal(0, 0.03), life))
    temp = (base_temp + 12 * (load - 0.7) + deg * severity * rng.uniform(10, 25)
            + rng.normal(0, 1.0, life) + np.linspace(0, rng.normal(0, 1.5), life))

    df = pd.DataFrame({"machine_id": machine_id, "cycle": cycles, "load": load,
                       "vibration_g": vib, "temperature_c": temp,
                       "rul": life - 1 - cycles, "failure_mode": mode,
                       "true_degradation": deg})       # hidden ground truth, never a feature
    for col in ("vibration_g", "temperature_c"):
        df.loc[rng.random(life) < 0.02, col] = np.nan
    return df


def simulate_fleet(n_machines: int = 40, min_life: int = 150, max_life: int = 350,
                   seed: int = 13) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    lives = rng.integers(min_life, max_life, n_machines)
    return pd.concat([simulate_machine(m, int(l), rng) for m, l in enumerate(lives)],
                     ignore_index=True)


def make_classification_labels(df: pd.DataFrame, horizon: int = 20) -> pd.DataFrame:
    """will_fail_soon = 1 when the machine fails within `horizon` cycles."""
    df = df.copy()
    df["will_fail_soon"] = (df["rul"] <= horizon).astype(int)
    return df


if __name__ == "__main__":
    fleet = make_classification_labels(simulate_fleet())
    out = Path(__file__).resolve().parents[1] / "data" / "sensor_log.csv"
    fleet.to_csv(out, index=False)
    print(f"{fleet.machine_id.nunique()} machines, {len(fleet)} readings -> {out}")
    print(fleet.groupby("machine_id").failure_mode.first().value_counts().to_string())
    print(f"positive rate {fleet.will_fail_soon.mean():.1%}, missing {fleet.vibration_g.isna().mean():.1%}")
