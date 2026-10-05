"""
Model comparison and alert evaluation with machine-grouped cross-validation.

Why grouped CV
--------------
Readings from the same machine are strongly correlated. If a machine's
early life is in the training set and its late life in the test set, the
model partly memorises that machine. Every split here is by machine_id
(GroupKFold), so test machines are never seen in training, and results
are averaged over 5 folds instead of a single lucky/unlucky split.

Two levels of evaluation
------------------------
1. Row level: ROC-AUC and average precision (PR-AUC; the better metric
   when only ~8% of readings are positive) for several models, including
   a no-model baseline (the raw vibration value used as the score).
2. Machine level, which is what maintenance teams care about: an alarm
   fires when the predicted probability exceeds a threshold for 2
   consecutive cycles. Each test machine ends in one of:
     caught       first alarm while the machine is degrading, >= 2 cycles before failure
     false alarm  first alarm while the machine is still healthy (simulator ground truth)
     missed       no alarm, or less than 2 cycles of warning
   The threshold is chosen *inside the training fold* (nested CV) to
   minimise 10 x missed + 3 x false alarms over the training machines.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import FEATURE_SETS, add_features
from simulate_sensor_data import make_classification_labels, simulate_fleet

HORIZON = 20            # label: fails within 20 cycles
MIN_WARNING = 2         # an alarm with < 2 cycles of warning is too late to act on
COST_MISS, COST_FALSE_ALARM = 10.0, 3.0


def make_models(seed: int = 0) -> dict:
    rf = dict(n_estimators=300, max_depth=8, min_samples_leaf=20, class_weight="balanced",
              random_state=seed, n_jobs=-1)
    return {
        "Raw vibration only (no model)": ("threshold", ["vibration_g"]),
        "Logistic regression, raw": (make_pipeline(StandardScaler(), LogisticRegression(
            class_weight="balanced", max_iter=2000)), FEATURE_SETS["raw"]),
        "Random forest, raw": (RandomForestClassifier(**rf), FEATURE_SETS["raw"]),
        "Logistic regression, engineered": (make_pipeline(StandardScaler(), LogisticRegression(
            class_weight="balanced", max_iter=2000)), FEATURE_SETS["engineered"]),
        "Random forest, engineered": (RandomForestClassifier(**rf), FEATURE_SETS["engineered"]),
    }


def _fit_predict(model, cols, train, test):
    if model == "threshold":
        return test[cols[0]].to_numpy()
    model.fit(train[cols], train["will_fail_soon"])
    return model.predict_proba(test[cols])[:, 1]


def _featurise_fold(raw: pd.DataFrame, train_ids, test_ids):
    train, coef = add_features(raw[raw.machine_id.isin(train_ids)])
    test, _ = add_features(raw[raw.machine_id.isin(test_ids)], load_coef=coef)
    return train, test


def machine_alarms(df: pd.DataFrame, prob: np.ndarray, threshold: float, consecutive: int = 2):
    """Per-machine outcome of the alarm rule (see module docstring)."""
    df = df.assign(prob=prob)
    rows = []
    for mid, g in df.groupby("machine_id"):
        above = (g["prob"].to_numpy() >= threshold).astype(int)
        run = np.convolve(above, np.ones(consecutive, int), "full")[:len(above)] >= consecutive
        life = int(g["cycle"].max()) + 1
        onset = int(g["cycle"].to_numpy()[np.argmax(g["true_degradation"].to_numpy() > 0)])
        if run.any():
            i = int(np.argmax(run))
            first = int(g["cycle"].to_numpy()[i])
            lead = life - 1 - first
            if g["true_degradation"].to_numpy()[i] == 0:
                outcome = "false alarm"
            else:
                outcome = "caught" if lead >= MIN_WARNING else "missed"
        else:
            lead, outcome = np.nan, "missed"
        rows.append({"machine_id": mid, "failure_mode": g["failure_mode"].iloc[0], "life": life,
                     "degradation_cycles": life - onset, "warning_cycles": lead, "outcome": outcome})
    return pd.DataFrame(rows)


def best_threshold(df: pd.DataFrame, p: np.ndarray) -> float:
    """Threshold minimising machine-level cost on (out-of-fold) training predictions."""
    grid = np.unique(np.quantile(p, np.linspace(0.5, 0.999, 120)))
    def cost(t):
        o = machine_alarms(df, p, t)["outcome"]
        return COST_MISS * (o == "missed").sum() + COST_FALSE_ALARM * (o == "false alarm").sum()
    return float(grid[int(np.argmin([cost(t) for t in grid]))])


def cross_validate(raw: pd.DataFrame | None = None, n_splits: int = 5, seed: int = 0):
    """Returns (row_metrics, machine_outcomes, oof_predictions)."""
    raw = make_classification_labels(simulate_fleet()) if raw is None else raw
    machines = raw["machine_id"].to_numpy()
    outer = GroupKFold(n_splits=n_splits)
    metrics, outcomes, oof = [], [], []
    for fold, (tr_idx, te_idx) in enumerate(outer.split(raw, groups=machines)):
        train_ids, test_ids = np.unique(machines[tr_idx]), np.unique(machines[te_idx])
        train, test = _featurise_fold(raw, train_ids, test_ids)
        inner = []                      # inner folds on training machines, featurised once
        for itr, ite in GroupKFold(4).split(train, groups=train["machine_id"]):
            a, b = _featurise_fold(raw, train["machine_id"].iloc[itr].unique(),
                                   train["machine_id"].iloc[ite].unique())
            inner.append((a, b, train.index.get_indexer(b.index)))
        y = test["will_fail_soon"].to_numpy()
        fold_oof = test.assign(fold=fold)
        for name, (model, cols) in make_models(seed).items():
            p = _fit_predict(model if model == "threshold" else clone(model), cols, train, test)
            metrics.append({"fold": fold, "model": name, "roc_auc": roc_auc_score(y, p),
                            "avg_precision": average_precision_score(y, p)})
            # Nested: choose the alarm threshold from out-of-fold predictions on training machines.
            inner_p = np.empty(len(train))
            for a, b, pos in inner:
                inner_p[pos] = _fit_predict(model if model == "threshold" else clone(model), cols, a, b)
            thr = best_threshold(train, inner_p)
            outcomes.append(machine_alarms(test, p, thr).assign(fold=fold, model=name, threshold=thr))
            fold_oof[f"prob::{name}"] = p
            fold_oof[f"thr::{name}"] = thr
        oof.append(fold_oof)
    return pd.DataFrame(metrics), pd.concat(outcomes, ignore_index=True), pd.concat(oof)


def permutation_importance_cv(raw: pd.DataFrame | None = None, n_splits: int = 5,
                              n_repeats: int = 5, seed: int = 0) -> pd.DataFrame:
    """Drop in held-out average precision when each feature is shuffled
    (random forest, engineered features), averaged over grouped folds."""
    from sklearn.inspection import permutation_importance
    raw = make_classification_labels(simulate_fleet()) if raw is None else raw
    model, cols = make_models(seed)["Random forest, engineered"]
    rows = []
    for tr, te in GroupKFold(n_splits).split(raw, groups=raw["machine_id"]):
        train, test = _featurise_fold(raw, raw.machine_id.iloc[tr].unique(), raw.machine_id.iloc[te].unique())
        m = clone(model).fit(train[cols], train["will_fail_soon"])
        r = permutation_importance(m, test[cols], test["will_fail_soon"], scoring="average_precision",
                                   n_repeats=n_repeats, random_state=seed, n_jobs=-1)
        rows.append(pd.Series(r.importances_mean, index=cols))
    return pd.DataFrame(rows).agg(["mean", "std"]).T.sort_values("mean", ascending=False)


def summarise(metrics: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    row = metrics.groupby("model")[["roc_auc", "avg_precision"]].agg(["mean", "std"])
    row.columns = [f"{a}_{b}" for a, b in row.columns]
    m = outcomes.groupby("model")
    row["machines_caught"] = m["outcome"].apply(lambda s: (s == "caught").mean())
    row["machines_missed"] = m["outcome"].apply(lambda s: (s == "missed").mean())
    row["false_alarms"] = m["outcome"].apply(lambda s: (s == "false alarm").mean())
    row["median_warning_cycles"] = outcomes[outcomes.outcome == "caught"].groupby("model")["warning_cycles"].median()
    return row.sort_values("avg_precision_mean", ascending=False)


if __name__ == "__main__":
    pd.set_option("display.width", 160)
    metrics, outcomes, _ = cross_validate()
    print(summarise(metrics, outcomes).round(3).to_string())
    best = outcomes[outcomes.model == "Random forest, engineered"]
    print("\nRandom forest (engineered) by failure mode:")
    print(pd.crosstab(best.failure_mode, best.outcome).to_string())
