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
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import GroupKFold
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

from .config import ALL_FEATURE_NAMES, SUBJECT_LIST, normalise_subject_id
from .data_processing import load_subject
from .persistence import load_obj, save_obj
from .splits import aggregate_by_video, video_leaveoneout, window_groupkfold


# ---------------------------------------------------------------- factories
ModelFactory = Callable[[], Any]

MODEL_FACTORIES: Dict[str, ModelFactory] = {
     "RandomForest": lambda: RandomForestRegressor(
        n_estimators=200, n_jobs=-1, random_state=42),
    "Ridge":  lambda: Ridge(alpha=10.0),
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

# High/low classification (threshold at 5) — reframes V/A prediction as the
# standard DEAP protocol, far more robust to single-rating-per-video label
# noise than exact 1-9 regression.
CLF_FACTORIES: Dict[str, ModelFactory] = {
    "LogisticRegression": lambda: MultiOutputClassifier(
        LogisticRegression(max_iter=1000)),
    "RandomForestClf": lambda: RandomForestClassifier(
        n_estimators=300, n_jobs=-1, random_state=42),
    "GradientBoostClf": lambda: MultiOutputClassifier(
        GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)),
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


def _metrics_clf(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """V / A separated accuracy + F1 for the high/low (threshold=5) reframing."""
    return {
        "acc_V": float(accuracy_score(y_true[:, 0], y_pred[:, 0])),
        "acc_A": float(accuracy_score(y_true[:, 1], y_pred[:, 1])),
        "f1_V":  float(f1_score(y_true[:, 0], y_pred[:, 0], zero_division=0)),
        "f1_A":  float(f1_score(y_true[:, 1], y_pred[:, 1], zero_division=0)),
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


def _load_subject_normalized(subject_id: str, per_subject_normalize: bool = True):
    """load_subject + optional per-subject z-scoring on that subject's own window
    stats (mean/std over their own ~2000+ windows, uses only their own X, no
    labels -> no leakage). Removes each person's absolute EEG/GSR scale offset
    before pooling across subjects, which is the standard cross-subject-invariance
    trick in EEG/BCI work (distinct from -- and complementary to -- the fold-level
    StandardScaler fit downstream on the pooled train split)."""
    sid = normalise_subject_id(subject_id)
    X, y, vids = load_subject(sid)
    if per_subject_normalize:
        X = StandardScaler().fit_transform(X)
    return sid, X, y, vids


# ---------------------------------------------------------------- 4a runner

def run_window_cv_pooled(
    subject_ids: Iterable[str],
    model_factory: ModelFactory,
    n_splits: int = 5,
    per_subject_normalize: bool = True,
) -> Dict[str, float]:

    X_parts, y_parts, group_parts = [], [], []

    for subject_id in subject_ids:
        sid, X, y, vids = _load_subject_normalized(subject_id, per_subject_normalize)
        group_id = int(sid) * 100 + vids
        X_parts.append(X)
        y_parts.append(y)
        group_parts.append(group_id)

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    groups = np.concatenate(group_parts, axis=0)

    fold_rows: List[Dict[str, float]] = []
    for tr, te in window_groupkfold(X, y, groups, n_splits=n_splits):
        sc = StandardScaler().fit(X[tr])
        mdl = model_factory().fit(sc.transform(X[tr]), y[tr])
        y_pred = mdl.predict(sc.transform(X[te]))
        m = _metrics(y[te], y_pred)
        m.update(_aggregate_window_to_video(y[te], y_pred, groups[te]))
        fold_rows.append(m)

    df = pd.DataFrame(fold_rows)
    return {k: float(df[k].mean()) for k in df.columns}



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

def batch_window_cv_pooled(
    subject_ids: Optional[Iterable[str]] = None,
    factories: Optional[Dict[str, ModelFactory]] = None,
) -> pd.DataFrame:
    subject_ids = list(subject_ids) if subject_ids is not None else SUBJECT_LIST
    factories = factories if factories is not None else MODEL_FACTORIES

    rows: List[Dict[str, Any]] = []
    for mname, fac in factories.items():
        metrics = run_window_cv_pooled(subject_ids, fac)
        rows.append({"subject": "pooled", "model": mname, **metrics})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- video-level pooled
def _pool_videos(subject_ids: Iterable[str], per_subject_normalize: bool = True):
    """Aggregate each subject to 40 video-level rows (aggregate_by_video), then
    pool across subjects. groups = subject id, so CV can hold out whole subjects
    -> tests generalization to unseen people, the realistic use case, instead of
    just unseen videos from people already in the training set.

    per_subject_normalize: z-score each subject's own WINDOW-level features (own
    mean/std, no labels used) before aggregating to video level -- removes each
    person's absolute physiological scale/offset prior to pooling, standard
    cross-subject-invariance trick for EEG. Uses window-level stats (thousands of
    samples/subject) rather than the 40 video-level rows for a stabler estimate."""
    Xv_parts, yv_parts, subj_parts = [], [], []
    for subject_id in subject_ids:
        sid, X, y, vids = _load_subject_normalized(subject_id, per_subject_normalize)
        Xv, yv, _ = aggregate_by_video(X, y, vids)
        Xv_parts.append(Xv)
        yv_parts.append(yv)
        subj_parts.append(np.full(len(Xv), int(sid)))
    Xv = np.concatenate(Xv_parts, axis=0)
    yv = np.concatenate(yv_parts, axis=0)
    subj_groups = np.concatenate(subj_parts, axis=0)
    return Xv, yv, subj_groups


def run_video_cv_pooled(
    subject_ids: Iterable[str],
    model_factory: ModelFactory,
    n_splits: int = 5,
) -> Dict[str, float]:
    """Video-level pooled regression CV across subjects (1280 rows max, one
    correctly-labeled row per video -- fixes the window/label mismatch that
    limits window-level CV, unlike single-subject LOVO gives many more samples)."""
    Xv, yv, subj_groups = _pool_videos(subject_ids)
    n_splits = min(n_splits, len(set(subj_groups.tolist())))

    fold_rows: List[Dict[str, float]] = []
    for tr, te in GroupKFold(n_splits=n_splits).split(Xv, yv, groups=subj_groups):
        sc = StandardScaler().fit(Xv[tr])
        mdl = model_factory().fit(sc.transform(Xv[tr]), yv[tr])
        y_pred = mdl.predict(sc.transform(Xv[te]))
        fold_rows.append(_metrics(yv[te], y_pred))

    df = pd.DataFrame(fold_rows)
    return {k: float(df[k].mean()) for k in df.columns}


def batch_video_cv_pooled(
    subject_ids: Optional[Iterable[str]] = None,
    factories: Optional[Dict[str, ModelFactory]] = None,
) -> pd.DataFrame:
    subject_ids = list(subject_ids) if subject_ids is not None else SUBJECT_LIST
    factories = factories if factories is not None else MODEL_FACTORIES

    rows: List[Dict[str, Any]] = []
    for mname, fac in factories.items():
        metrics = run_video_cv_pooled(subject_ids, fac)
        rows.append({"subject": "pooled_video", "model": mname, **metrics})
    return pd.DataFrame(rows)


def run_video_cv_pooled_clf(
    subject_ids: Iterable[str],
    model_factory: ModelFactory,
    n_splits: int = 5,
    threshold: float = 5.0,
    per_subject_threshold: bool = True,
) -> Dict[str, float]:
    """Same pooled video-level split as run_video_cv_pooled, but V/A binarized
    and scored with accuracy/F1 instead of R2 -- much more robust to
    single-rating label noise than exact regression.

    per_subject_threshold: split each subject's own 40 videos at THEIR OWN
    median V/A instead of a fixed global `threshold`. DEAP self-reports carry a
    lot of personal rating bias (some people never rate above 6, others rarely
    go below 4) -- a fixed threshold=5 pooled across 32 subjects here gives an
    ~22-25% positive rate (checked directly), so a majority-class predictor gets
    ~75-78% accuracy for free with ~0 F1. Per-subject median split uses only
    that subject's own labels (each subject sits entirely in train or entirely
    in test per fold, so no cross-subject leakage) and yields a ~50/50 balance."""
    Xv, yv_cont, subj_groups = _pool_videos(subject_ids)
    if per_subject_threshold:
        yv = np.zeros_like(yv_cont, dtype=int)
        for sid_val in np.unique(subj_groups):
            mask = subj_groups == sid_val
            med = np.median(yv_cont[mask], axis=0)
            yv[mask] = (yv_cont[mask] >= med).astype(int)
    else:
        yv = (yv_cont >= threshold).astype(int)
    n_splits = min(n_splits, len(set(subj_groups.tolist())))

    fold_rows: List[Dict[str, float]] = []
    for tr, te in GroupKFold(n_splits=n_splits).split(Xv, yv, groups=subj_groups):
        sc = StandardScaler().fit(Xv[tr])
        mdl = model_factory().fit(sc.transform(Xv[tr]), yv[tr])
        y_pred = mdl.predict(sc.transform(Xv[te]))
        fold_rows.append(_metrics_clf(yv[te], y_pred))

    df = pd.DataFrame(fold_rows)
    return {k: float(df[k].mean()) for k in df.columns}


def batch_video_cv_pooled_clf(
    subject_ids: Optional[Iterable[str]] = None,
    factories: Optional[Dict[str, ModelFactory]] = None,
) -> pd.DataFrame:
    subject_ids = list(subject_ids) if subject_ids is not None else SUBJECT_LIST
    factories = factories if factories is not None else CLF_FACTORIES

    rows: List[Dict[str, Any]] = []
    for mname, fac in factories.items():
        metrics = run_video_cv_pooled_clf(subject_ids, fac)
        rows.append({"subject": "pooled_video_clf", "model": mname, **metrics})
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
