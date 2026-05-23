"""Classical ML for [Valence, Arousal] regression — stage 5 (classical branch).

Provides:
- MODEL_FACTORIES : window-level models (RandomForest, GradientBoost).
- LOVO_FACTORIES  : video-level models (Dummy, Ridge, RandomForest).
- run_window_cv(subject_id, factory) : window-level 5-fold GroupKFold, returns
  a row of mean metrics (mae/rmse/r2 for V and A, plus video-aggregated).
- run_lovo(subject_id) : runs all LOVO_FACTORIES, returns one row per model.
- batch_window_cv / batch_lovo : iterate subjects -> DataFrame of metrics.
- fit_and_save_final / load_subject_model : persist one trained estimator per
  subject for later inference.

Functions are I/O free except for save/load via persistence helpers, so they can
be smoke-tested with synthetic data.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

from .config import ALL_FEATURE_NAMES, SUBJECT_LIST, normalise_subject_id
from .data_processing import load_subject
from .persistence import load_obj, save_obj
from .splits import aggregate_by_video, video_leaveoneout, window_groupkfold


# ---------------------------------------------------------------- factories
ModelFactory = Callable[[], Any]

MODEL_FACTORIES: Dict[str, ModelFactory] = {
    "RandomForest":  lambda: RandomForestRegressor(
        n_estimators=200, n_jobs=-1, random_state=42),
    "GradientBoost": lambda: MultiOutputRegressor(
        GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)),
}

LOVO_FACTORIES: Dict[str, ModelFactory] = {
    "Dummy":        lambda: DummyRegressor(strategy="mean"),
    "Ridge":        lambda: Ridge(alpha=10.0),
    "RandomForest": lambda: RandomForestRegressor(
        n_estimators=300, n_jobs=-1, random_state=42),
}


# ---------------------------------------------------------------- metrics
def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """V / A separated metrics. Always returns 6 floats."""
    mae  = mean_absolute_error(y_true, y_pred, multioutput="raw_values")
    rmse = root_mean_squared_error(y_true, y_pred, multioutput="raw_values")
    r2   = r2_score(y_true, y_pred, multioutput="raw_values")
    return {
        "mae_V":  float(mae[0]),  "mae_A":  float(mae[1]),
        "rmse_V": float(rmse[0]), "rmse_A": float(rmse[1]),
        "r2_V":   float(r2[0]),   "r2_A":   float(r2[1]),
    }


def _aggregate_window_to_video(
    y_true: np.ndarray, y_pred: np.ndarray, vids: np.ndarray,
) -> Dict[str, float]:
    """Median of window predictions per video, compared to first video label."""
    df = pd.DataFrame({
        "vid": vids,
        "V_true": y_true[:, 0], "A_true": y_true[:, 1],
        "V_pred": y_pred[:, 0], "A_pred": y_pred[:, 1],
    })
    g = df.groupby("vid").agg(
        V_true=("V_true", "first"), A_true=("A_true", "first"),
        V_pred=("V_pred", "median"), A_pred=("A_pred", "median"))
    yt = g[["V_true", "A_true"]].values
    yp = g[["V_pred", "A_pred"]].values
    out = _metrics(yt, yp)
    return {f"vid_{k}": v for k, v in out.items()}


# ---------------------------------------------------------------- 4a runner
def run_window_cv(
    subject_id: str,
    model_factory: ModelFactory,
    n_splits: int = 5,
) -> Dict[str, float]:
    """Window-level GroupKFold CV for one subject. Returns averaged metrics."""
    sid = normalise_subject_id(subject_id)
    X, y, vids = load_subject(sid)

    fold_rows: List[Dict[str, float]] = []
    for tr, te in window_groupkfold(X, y, vids, n_splits=n_splits):
        sc = StandardScaler().fit(X[tr])
        mdl = model_factory().fit(sc.transform(X[tr]), y[tr])
        y_pred = mdl.predict(sc.transform(X[te]))
        m = _metrics(y[te], y_pred)
        m.update(_aggregate_window_to_video(y[te], y_pred, vids[te]))
        fold_rows.append(m)

    df = pd.DataFrame(fold_rows)
    return {k: float(df[k].mean()) for k in df.columns}


# ---------------------------------------------------------------- 4b runner
def run_lovo(subject_id: str) -> List[Dict[str, Any]]:
    """Video-level LeaveOneVideoOut for all LOVO_FACTORIES. One row per model."""
    sid = normalise_subject_id(subject_id)
    X, y, vids = load_subject(sid)
    Xv, yv, groups = aggregate_by_video(X, y, vids)

    out: List[Dict[str, Any]] = []
    for mname, factory in LOVO_FACTORIES.items():
        preds = np.zeros_like(yv)
        for tr, te in video_leaveoneout(Xv, yv, groups):
            sc = StandardScaler().fit(Xv[tr])
            preds[te] = factory().fit(sc.transform(Xv[tr]), yv[tr]) \
                                .predict(sc.transform(Xv[te]))
        row: Dict[str, Any] = {"subject": sid, "model": mname}
        row.update(_metrics(yv, preds))
        out.append(row)
    return out


# ---------------------------------------------------------------- batch
def batch_window_cv(
    subject_ids: Optional[Iterable[str]] = None,
    factories: Optional[Dict[str, ModelFactory]] = None,
) -> pd.DataFrame:
    """Iterate (subjects, models) -> tidy DataFrame of mean metrics."""
    subject_ids = list(subject_ids) if subject_ids is not None else SUBJECT_LIST
    factories = factories if factories is not None else MODEL_FACTORIES

    rows: List[Dict[str, Any]] = []
    for sid in subject_ids:
        sid = normalise_subject_id(sid)
        for mname, fac in factories.items():
            metrics = run_window_cv(sid, fac)
            rows.append({"subject": sid, "model": mname, **metrics})
    return pd.DataFrame(rows)


def batch_lovo(subject_ids: Optional[Iterable[str]] = None) -> pd.DataFrame:
    subject_ids = list(subject_ids) if subject_ids is not None else SUBJECT_LIST
    rows: List[Dict[str, Any]] = []
    for sid in subject_ids:
        rows.extend(run_lovo(sid))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- persistence
def fit_and_save_final(
    subject_id: str, model_name: str, level: str = "window",
    force: bool = False,
) -> str:
    """Fit (scaler, model) on ALL data of the subject and joblib-dump them."""
    sid = normalise_subject_id(subject_id)
    tag = f"{'win' if level == 'window' else 'vid'}_{model_name}_s{sid}"
    if not force and load_obj(tag) is not None:
        return tag

    X, y, vids = load_subject(sid)
    if level == "video":
        X, y, _ = aggregate_by_video(X, y, vids)

    factories = MODEL_FACTORIES if level == "window" else LOVO_FACTORIES
    if model_name not in factories:
        raise KeyError(f"Unknown model '{model_name}' for level '{level}'")

    sc = StandardScaler().fit(X)
    mdl = factories[model_name]().fit(sc.transform(X), y)
    save_obj({"scaler": sc, "model": mdl,
              "features": ALL_FEATURE_NAMES, "level": level}, tag)
    return tag


def load_subject_model(
    subject_id: str, model_name: str, level: str = "window",
) -> Optional[Dict[str, Any]]:
    sid = normalise_subject_id(subject_id)
    tag = f"{'win' if level == 'window' else 'vid'}_{model_name}_s{sid}"
    return load_obj(tag)


def predict_va(
    subject_id: str, X: np.ndarray,
    model_name: str = "RandomForest", level: str = "window",
) -> np.ndarray:
    """Predict [V, A] from features using a previously fitted+saved model."""
    bundle = load_subject_model(subject_id, model_name, level=level)
    if bundle is None:
        raise FileNotFoundError(
            f"No saved model for s{subject_id} / {model_name} / {level}. "
            "Call fit_and_save_final() first.")
    return bundle["model"].predict(bundle["scaler"].transform(X))
