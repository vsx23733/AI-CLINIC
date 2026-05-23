"""Liking model + response surface — stages 7 (M2) and 8.

Pipeline:
1. load_deap_labels()             -> (40 trials/subject) Valence, Arousal, Liking
2. fit_liking_model(...)          -> g: (V, A) -> Liking  (pool across subjects)
3. response_surface(model, grid)  -> 2D heatmap over the (V, A) plane
4. optimal_zone(surface)          -> (V*, A*) = argmax + neighbourhood mask

We pool across subjects by default. Pass `subject_id=...` to fit a per-subject
model when you have enough trials (here 40 — borderline; pooled is more robust).

Visualisation is optional (matplotlib lazy-imported).
"""
from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import PolynomialFeatures

from .config import DATA_PATH, N_VIDEOS, SUBJECT_LIST, normalise_subject_id


# ---------------------------------------------------------------- DEAP labels
def load_deap_labels(
    subject_ids: Optional[Iterable[str]] = None,
    data_path: str = DATA_PATH,
) -> pd.DataFrame:
    """Returns a long DataFrame with columns:
    subject, video, Valence, Arousal, Dominance, Liking.
    One row per (subject, video) -> 32 * 40 = 1280 rows max.
    """
    subject_ids = list(subject_ids) if subject_ids is not None else SUBJECT_LIST
    rows = []
    for sid in subject_ids:
        sid = normalise_subject_id(sid)
        path = os.path.join(data_path, f"s{sid}.dat")
        if not os.path.exists(path):
            continue
        with open(path, "rb") as f:
            sub = pickle.load(f, encoding="latin1")
        labels = sub["labels"]                        # (40, 4)
        for v in range(min(N_VIDEOS, len(labels))):
            rows.append({
                "subject": sid, "video": v,
                "Valence":   float(labels[v, 0]),
                "Arousal":   float(labels[v, 1]),
                "Dominance": float(labels[v, 2]),
                "Liking":    float(labels[v, 3]),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- fit
@dataclass
class LikingModel:
    """Thin wrapper exposing predict(V, A) -> Liking estimate."""
    model: object
    kind:  str                    # 'kernel_ridge' | 'rf' | 'poly2'
    extras: dict

    def predict(self, V: np.ndarray, A: np.ndarray) -> np.ndarray:
        V = np.asarray(V, dtype=np.float32).ravel()
        A = np.asarray(A, dtype=np.float32).ravel()
        X = np.column_stack([V, A])
        if self.kind == "poly2":
            X = self.extras["poly"].transform(X)
        return self.model.predict(X)


def fit_liking_model(
    df_labels: pd.DataFrame,
    kind: str = "kernel_ridge",
) -> LikingModel:
    """Fit g: (V, A) -> Liking on the labels frame.

    kind:
      - 'kernel_ridge' : smooth non-linear surface (recommended default).
      - 'rf'           : RandomForest, captures non-monotonic effects.
      - 'poly2'        : quadratic surface, interpretable closed form.
    """
    X = df_labels[["Valence", "Arousal"]].values.astype(np.float32)
    y = df_labels["Liking"].values.astype(np.float32)

    if kind == "kernel_ridge":
        mdl = KernelRidge(alpha=1.0, kernel="rbf", gamma=0.1).fit(X, y)
        return LikingModel(mdl, kind, {})
    if kind == "rf":
        mdl = RandomForestRegressor(
            n_estimators=300, n_jobs=-1, random_state=42).fit(X, y)
        return LikingModel(mdl, kind, {})
    if kind == "poly2":
        poly = PolynomialFeatures(degree=2, include_bias=False)
        Xp = poly.fit_transform(X)
        from sklearn.linear_model import Ridge
        mdl = Ridge(alpha=1.0).fit(Xp, y)
        return LikingModel(mdl, kind, {"poly": poly})
    raise ValueError(f"Unknown kind='{kind}'")


# ---------------------------------------------------------------- response surface
@dataclass
class ResponseSurface:
    V_grid: np.ndarray            # (n_V,)
    A_grid: np.ndarray            # (n_A,)
    Liking: np.ndarray            # (n_V, n_A)

    @property
    def shape(self) -> Tuple[int, int]:
        return self.Liking.shape


def response_surface(
    model: LikingModel,
    v_range: Tuple[float, float] = (1.0, 9.0),
    a_range: Tuple[float, float] = (1.0, 9.0),
    step: float = 0.1,
) -> ResponseSurface:
    """Evaluate g(V, A) on a regular grid -> heatmap. Defaults to DEAP scale 1..9."""
    V = np.arange(v_range[0], v_range[1] + 1e-9, step, dtype=np.float32)
    A = np.arange(a_range[0], a_range[1] + 1e-9, step, dtype=np.float32)
    VV, AA = np.meshgrid(V, A, indexing="ij")           # (n_V, n_A)
    L = model.predict(VV.ravel(), AA.ravel()).reshape(VV.shape).astype(np.float32)
    return ResponseSurface(V_grid=V, A_grid=A, Liking=L)


# ---------------------------------------------------------------- optimal zone
@dataclass
class OptimalZone:
    V_star: float
    A_star: float
    Liking_star: float
    zone_mask: np.ndarray                # (n_V, n_A) boolean
    threshold: float                     # liking >= threshold defines the zone


def optimal_zone(
    surface: ResponseSurface, top_pct: float = 0.10,
) -> OptimalZone:
    """Argmax + a connected high-liking region.

    `top_pct` = fraction of the surface mass considered 'high liking' (default
    top 10 %). Useful to advise within a *zone*, not just a single point.
    """
    L = surface.Liking
    flat_idx = int(np.argmax(L))
    iV, iA = np.unravel_index(flat_idx, L.shape)
    V_star = float(surface.V_grid[iV])
    A_star = float(surface.A_grid[iA])
    Liking_star = float(L[iV, iA])

    thr = float(np.quantile(L, 1.0 - top_pct))
    mask = L >= thr
    return OptimalZone(V_star, A_star, Liking_star, mask, thr)


# ---------------------------------------------------------------- viz (optional)
def plot_surface(surface: ResponseSurface, zone: Optional[OptimalZone] = None):
    """Quick heatmap. Returns the matplotlib Axes. Lazy-imports matplotlib."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    extent = [surface.A_grid[0], surface.A_grid[-1],
              surface.V_grid[0], surface.V_grid[-1]]
    im = ax.imshow(surface.Liking, origin="lower", aspect="auto",
                   extent=extent, cmap="viridis")
    ax.set_xlabel("Arousal"); ax.set_ylabel("Valence")
    ax.set_title("Liking response surface  g(V, A)")
    plt.colorbar(im, ax=ax, label="Predicted Liking")
    if zone is not None:
        ax.contour(surface.A_grid, surface.V_grid, zone.zone_mask.astype(int),
                   levels=[0.5], colors="white", linewidths=1.5)
        ax.scatter([zone.A_star], [zone.V_star], color="red",
                   marker="*", s=180, label=f"(V*={zone.V_star:.1f}, A*={zone.A_star:.1f})")
        ax.legend(loc="lower left")
    return ax
