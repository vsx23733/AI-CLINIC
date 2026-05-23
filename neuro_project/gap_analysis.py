"""Gap analysis — stage 9.

Takes predicted (V, A) from M1 (physio model) and compares to the optimal zone
from M2 (Liking surface). Produces:
  - GapReport dataclass with numerical fields
  - per-window/per-segment weak spots
  - a structured text summary ready for the downstream LLM prompt builder

No I/O. Pure functions over arrays + the LikingModel/OptimalZone objects.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .liking_model import LikingModel, OptimalZone


# ---------------------------------------------------------------- per-window
@dataclass
class WindowSegment:
    start_s: float
    end_s:   float
    V_hat:   float
    A_hat:   float
    Liking_pred: float

    def to_dict(self) -> dict:
        return {"start_s": self.start_s, "end_s": self.end_s,
                "V_hat": self.V_hat, "A_hat": self.A_hat,
                "Liking_pred": self.Liking_pred}


def per_window_liking(
    V_hat: np.ndarray, A_hat: np.ndarray, liking_model: LikingModel,
    window_size_s: float = 2.0, step_s: float = 0.125,
) -> List[WindowSegment]:
    """One WindowSegment per prediction. Times assume contiguous sliding windows
    starting at t=0; adjust if your ad has trimming/baseline."""
    V_hat = np.asarray(V_hat).ravel()
    A_hat = np.asarray(A_hat).ravel()
    L = liking_model.predict(V_hat, A_hat)
    out: List[WindowSegment] = []
    for i, (v, a, l) in enumerate(zip(V_hat, A_hat, L)):
        t0 = i * step_s
        out.append(WindowSegment(t0, t0 + window_size_s,
                                 float(v), float(a), float(l)))
    return out


def weak_segments(
    segments: List[WindowSegment], k: int = 5,
) -> List[WindowSegment]:
    """k segments with the lowest predicted Liking."""
    return sorted(segments, key=lambda s: s.Liking_pred)[:k]


# ---------------------------------------------------------------- ad-level report
@dataclass
class GapReport:
    # measured
    V_hat: float
    A_hat: float
    Liking_pred: float
    # target
    V_star: float
    A_star: float
    Liking_star: float
    # derived
    dV: float                                # V_star - V_hat
    dA: float                                # A_star - A_hat
    in_zone: bool                            # within high-liking mask
    distance: float                          # Euclidean (V,A) gap
    # context
    n_windows: int = 0
    weak: List[WindowSegment] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "V_hat": self.V_hat, "A_hat": self.A_hat, "Liking_pred": self.Liking_pred,
            "V_star": self.V_star, "A_star": self.A_star, "Liking_star": self.Liking_star,
            "dV": self.dV, "dA": self.dA, "in_zone": self.in_zone,
            "distance": self.distance, "n_windows": self.n_windows,
            "weak": [w.to_dict() for w in self.weak],
        }


def _is_in_zone(V: float, A: float, zone: OptimalZone,
                V_grid: np.ndarray, A_grid: np.ndarray) -> bool:
    iV = int(np.argmin(np.abs(V_grid - V)))
    iA = int(np.argmin(np.abs(A_grid - A)))
    return bool(zone.zone_mask[iV, iA])


def compute_gap(
    V_hat: np.ndarray, A_hat: np.ndarray,
    liking_model: LikingModel,
    zone: OptimalZone,
    V_grid: np.ndarray, A_grid: np.ndarray,
    weak_k: int = 5,
    window_size_s: float = 2.0, step_s: float = 0.125,
) -> GapReport:
    """Single ad-level report. V_hat/A_hat can be 1D (per-window) or scalar."""
    V_arr = np.asarray(V_hat, dtype=np.float32).ravel()
    A_arr = np.asarray(A_hat, dtype=np.float32).ravel()
    if V_arr.size == 0:
        raise ValueError("Empty V_hat/A_hat")

    # ad-level summary = median across windows (robust to outliers)
    V_m = float(np.median(V_arr))
    A_m = float(np.median(A_arr))
    L_m = float(liking_model.predict([V_m], [A_m])[0])

    segs = per_window_liking(V_arr, A_arr, liking_model,
                             window_size_s=window_size_s, step_s=step_s)
    weak = weak_segments(segs, k=weak_k) if V_arr.size > 1 else []

    return GapReport(
        V_hat=V_m, A_hat=A_m, Liking_pred=L_m,
        V_star=zone.V_star, A_star=zone.A_star, Liking_star=zone.Liking_star,
        dV=zone.V_star - V_m, dA=zone.A_star - A_m,
        in_zone=_is_in_zone(V_m, A_m, zone, V_grid, A_grid),
        distance=float(np.hypot(zone.V_star - V_m, zone.A_star - A_m)),
        n_windows=int(V_arr.size), weak=weak,
    )


# ---------------------------------------------------------------- textual report
def format_report(
    report: GapReport, ad_description: Optional[str] = None,
) -> str:
    """Human-readable, LLM-ready text. Plug `ad_description` in for the prompt
    builder of the LLM advisor stage."""
    lines = [
        "PREDICTED RESPONSE",
        f"  Valence: {report.V_hat:.2f}/9   Arousal: {report.A_hat:.2f}/9   "
        f"Predicted Liking: {report.Liking_pred:.2f}/9",
        "TARGET ZONE (max Liking)",
        f"  Valence*: {report.V_star:.2f}    Arousal*: {report.A_star:.2f}    "
        f"Liking*: {report.Liking_star:.2f}/9",
        "GAP",
        f"  dV: {report.dV:+.2f}   dA: {report.dA:+.2f}   "
        f"euclidean: {report.distance:.2f}   in_zone: {report.in_zone}",
    ]
    if report.weak:
        lines.append(f"WEAK SEGMENTS (top {len(report.weak)} lowest liking)")
        for w in report.weak:
            lines.append(
                f"  {w.start_s:5.2f}s - {w.end_s:5.2f}s : "
                f"V={w.V_hat:.2f}  A={w.A_hat:.2f}  L={w.Liking_pred:.2f}")
    if ad_description:
        lines += ["AD DESCRIPTION (external metadata, not from DEAP)",
                  f"  {ad_description}"]
    return "\n".join(lines)
