"""Save/load helpers for the pipeline.

- DataFrames -> pickle (preserves dtypes) + a side-by-side CSV for readability.
- sklearn estimators / scalers -> joblib (compressed).
- PyTorch checkpoints -> torch.save (lazy import; skipped if torch unavailable).

All functions are None-safe (load returns None if the file is missing) so callers
can write conditional 'resume' logic without try/except boilerplate.
"""
from __future__ import annotations

import os
from typing import Any, Optional

import joblib
import pandas as pd

from .config import ARTIFACTS_DIR, MODELS_DIR


# ---------------------------------------------------------------- DataFrames
def save_df(df: pd.DataFrame, name: str, dirpath: str = ARTIFACTS_DIR) -> str:
    """Pickle the DataFrame and write a CSV next to it. Returns the .pkl path."""
    os.makedirs(dirpath, exist_ok=True)
    p_pkl = os.path.join(dirpath, name + ".pkl")
    p_csv = os.path.join(dirpath, name + ".csv")
    df.to_pickle(p_pkl)
    df.to_csv(p_csv, index=False)
    return p_pkl


def load_df(name: str, dirpath: str = ARTIFACTS_DIR) -> Optional[pd.DataFrame]:
    p = os.path.join(dirpath, name + ".pkl")
    return pd.read_pickle(p) if os.path.exists(p) else None


# ---------------------------------------------------------------- sklearn / generic
def save_obj(obj: Any, name: str, dirpath: str = MODELS_DIR) -> str:
    os.makedirs(dirpath, exist_ok=True)
    p = os.path.join(dirpath, name + ".joblib")
    joblib.dump(obj, p, compress=3)
    return p


def load_obj(name: str, dirpath: str = MODELS_DIR) -> Optional[Any]:
    p = os.path.join(dirpath, name + ".joblib")
    return joblib.load(p) if os.path.exists(p) else None


def list_artifacts(dirpath: str = ARTIFACTS_DIR) -> list[str]:
    """Quick inventory of what's currently persisted."""
    out: list[str] = []
    for root, _, files in os.walk(dirpath):
        for f in files:
            out.append(os.path.relpath(os.path.join(root, f), dirpath))
    return sorted(out)


# ---------------------------------------------------------------- torch
def save_torch(state_dict: Any, name: str, dirpath: str = MODELS_DIR) -> str:
    """Save a torch state_dict. Imports torch lazily."""
    import torch  # type: ignore
    os.makedirs(dirpath, exist_ok=True)
    p = os.path.join(dirpath, name + ".pt")
    torch.save(state_dict, p)
    return p


def load_torch(name: str, dirpath: str = MODELS_DIR, map_location: str = "cpu") -> Optional[Any]:
    p = os.path.join(dirpath, name + ".pt")
    if not os.path.exists(p):
        return None
    import torch  # type: ignore
    return torch.load(p, map_location=map_location)
