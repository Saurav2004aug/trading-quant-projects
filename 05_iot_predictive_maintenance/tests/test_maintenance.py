import numpy as np
import pandas as pd

from features import FEATURE_SETS, add_features
from simulate_sensor_data import make_classification_labels, simulate_fleet
from train_model import best_threshold, cross_validate, machine_alarms
from vibration_features import simulated_burst, vibration_features


def small_fleet():
    return make_classification_labels(simulate_fleet(n_machines=12, seed=3))


# ---------------------------------------------------------- firmware maths
def test_sine_rms_and_crest():
    f = vibration_features(simulated_burst(amp_g=0.2, n=1000, freq=50))   # whole periods
    assert abs(f["rms_g"] - 0.2 / np.sqrt(2)) < 1e-3
    assert abs(f["crest"] - np.sqrt(2)) < 1e-2
    assert abs(f["kurtosis"] - 1.5) < 1e-2


def test_gravity_orientation_does_not_matter():
    a = vibration_features(simulated_burst(gravity_axis=(0, 0, 1)))
    b = vibration_features(simulated_burst(gravity_axis=(1, 1, 0.2)))
    assert abs(a["rms_g"] - b["rms_g"]) < 1e-9


def test_impulses_raise_crest_and_kurtosis_before_rms():
    clean = vibration_features(simulated_burst(noise_g=0.01))
    faulty = vibration_features(simulated_burst(noise_g=0.01, impulses=10))
    assert faulty["crest"] > 1.5 * clean["crest"]
    assert faulty["kurtosis"] > 1.5 * clean["kurtosis"]
    assert faulty["rms_g"] < 1.2 * clean["rms_g"]


# ---------------------------------------------------------- data + features
def test_labels_and_ground_truth():
    df = small_fleet()
    last = df.groupby("machine_id").tail(21)
    assert (last.will_fail_soon == 1).all()
    assert df.groupby("machine_id").will_fail_soon.sum().eq(21).all()
    assert (df.groupby("machine_id").true_degradation.first() == 0).all()   # healthy at start
    assert (df.groupby("machine_id").true_degradation.last() > 0.8).all()   # worn out at the end


def test_features_are_causal():
    df = small_fleet()
    full, coef = add_features(df)
    m = df[df.machine_id == 0]
    cut = 80
    truncated, _ = add_features(m[m.cycle <= cut], load_coef=coef)
    cols = FEATURE_SETS["engineered"]
    a = full[(full.machine_id == 0) & (full.cycle <= cut)][cols].to_numpy()
    b = truncated[cols].to_numpy()
    assert np.allclose(a, b, equal_nan=True)


def test_no_missing_values_after_featurisation():
    df, _ = add_features(small_fleet())
    assert not df[FEATURE_SETS["engineered"]].isna().any().any()


# ---------------------------------------------------------- evaluation logic
def _toy_machine(prob, degr_from):
    n = len(prob)
    return pd.DataFrame({"machine_id": 0, "cycle": np.arange(n), "failure_mode": "gradual",
                         "true_degradation": (np.arange(n) >= degr_from).astype(float)}), np.array(prob)


def test_alarm_outcomes():
    df, p = _toy_machine([0, 0, 0, 0, 1, 1, 1, 1, 1, 1], degr_from=3)
    assert machine_alarms(df, p, 0.5).outcome[0] == "caught"
    df, p = _toy_machine([0, 1, 1, 0, 0, 0, 0, 0, 0, 0], degr_from=5)
    assert machine_alarms(df, p, 0.5).outcome[0] == "false alarm"
    df, p = _toy_machine([0, 1, 0, 1, 0, 0, 0, 0, 0, 0], degr_from=0)   # never 2 in a row
    assert machine_alarms(df, p, 0.5).outcome[0] == "missed"
    df, p = _toy_machine([0] * 9 + [1], degr_from=5)                     # too late
    assert machine_alarms(df, p, 0.5).outcome[0] == "missed"


def test_threshold_prefers_catching_failures():
    df, p = _toy_machine(list(np.linspace(0, 1, 30)), degr_from=20)
    thr = best_threshold(df, p)
    assert machine_alarms(df, p, thr).outcome[0] == "caught"


def test_grouped_cv_never_shares_machines():
    raw = small_fleet()
    metrics, outcomes, oof = cross_validate(raw, n_splits=3)
    per_fold = oof.groupby("fold").machine_id.apply(set)
    for i in per_fold.index:
        for j in per_fold.index:
            if i < j:
                assert not (per_fold[i] & per_fold[j])
    assert set(oof.machine_id) == set(raw.machine_id)
    # every model evaluated on every fold; outcomes cover every machine once per model
    assert metrics.groupby("model").size().eq(3).all()
    assert outcomes.groupby("model").machine_id.nunique().eq(12).all()
