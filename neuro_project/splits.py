"""Cross-validation splits for the pipeline.

- 4a: window-level GroupKFold (groups = video_id) -> stable per-subject metrics.
- 4b: video-level LeaveOneVideoOut on per-video aggregates -> honest framing,
       reflects real problem size (1 label per video on DEAP).

Both helpers are pure generators over (train_idx, test_idx); callers handle
scaling and fitting. Includes a verification function to assert no group leak.
"""
from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

from .config import ALL_FEATURE_NAMES


# ---------------------------------------------------------------- 4a window-level
def window_groupkfold(
    X: np.ndarray, y: np.ndarray, vids: np.ndarray, n_splits: int = 5,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, test_idx) for each fold. groups = vids ensures no video
    is split across train/test (windows of one video are correlated)."""
    gkf = GroupKFold(n_splits=n_splits)
    yield from gkf.split(X, y, groups=vids)


# ---------------------------------------------------------------- 4b video-level
def aggregate_by_video(
    X: np.ndarray, y: np.ndarray, vids: np.ndarray,
    feature_names: list[str] | None = None,
    agg: str = "mean",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collapse per-window features into per-video features.
    Returns (Xv, yv, groups) with Xv shape (n_videos, n_features)
    and yv shape (n_videos, n_targets). `groups` = sorted unique video ids,
    aligned with rows of Xv/yv.

    agg: 'mean' (default) or 'median'.
    """
    if feature_names is None:
        feature_names = ALL_FEATURE_NAMES
    df = pd.DataFrame(X, columns=feature_names)
    df["vid"] = vids
    if agg == "mean":
        Xv = df.groupby("vid")[feature_names].mean().values
    elif agg == "median":
        Xv = df.groupby("vid")[feature_names].median().values
    else:
        raise ValueError(f"Unknown agg='{agg}', use 'mean' or 'median'.")

    # targets : (n,2) or (n,k). Take first label per video (constant per video).
    n_targets = 1 if y.ndim == 1 else y.shape[1]
    cols = [f"_t{i}" for i in range(n_targets)]
    yy = y.reshape(-1, n_targets)
    df_y = pd.DataFrame(yy, columns=cols)
    df_y["vid"] = vids
    yv = df_y.groupby("vid").first()[cols].values

    groups = np.sort(np.unique(vids))
    return Xv.astype(np.float32), yv.astype(np.float32), groups


def video_leaveoneout(
    Xv: np.ndarray, yv: np.ndarray, groups: np.ndarray,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """LeaveOneGroupOut at video level. With 40 unique groups -> 40 folds,
    each test set = 1 video."""
    logo = LeaveOneGroupOut()
    yield from logo.split(Xv, yv, groups=groups)


# ---------------------------------------------------------------- safety check
def assert_no_group_leak(
    train_idx: np.ndarray, test_idx: np.ndarray, groups: np.ndarray,
) -> None:
    """Verify that no group appears in both train and test. Raises AssertionError
    if a leak is detected. Use once in dev, remove for production."""
    common = set(groups[train_idx]) & set(groups[test_idx])
    assert not common, f"Group leak detected: {sorted(common)}"
