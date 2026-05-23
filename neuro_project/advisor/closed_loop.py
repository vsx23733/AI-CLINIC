"""Stage 13 — Closed loop : compare an old GapReport (before LLM edits) against
a new one (after re-editing the ad and re-measuring the viewer/panel).

This is the A/B-style objective evaluation of the LLM recommendation. It does
NOT call the LLM — it just diffs two reports and tells you whether things got
better, worse, or stayed flat.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from ..gap_analysis import GapReport, WindowSegment


# ---------------------------------------------------------------- summary
@dataclass
class ImprovementSummary:
    # ad-level deltas (new - old)
    d_V_hat:        float
    d_A_hat:        float
    d_Liking_pred:  float
    d_distance:     float          # negative = closer to (V*, A*) = better
    in_zone_before: bool
    in_zone_after:  bool
    transition:    str             # 'entered_zone' | 'left_zone' | 'stayed_in' | 'stayed_out'
    verdict:       str             # 'improved' | 'unchanged' | 'regressed'
    # per-segment matches (if both reports had weak segments)
    weak_changes:  List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "d_V_hat": self.d_V_hat, "d_A_hat": self.d_A_hat,
            "d_Liking_pred": self.d_Liking_pred, "d_distance": self.d_distance,
            "in_zone_before": self.in_zone_before, "in_zone_after": self.in_zone_after,
            "transition": self.transition, "verdict": self.verdict,
            "weak_changes": self.weak_changes,
        }


# ---------------------------------------------------------------- helpers
def _transition(before: bool, after: bool) -> str:
    if before and after:       return "stayed_in"
    if not before and not after: return "stayed_out"
    return "entered_zone" if after else "left_zone"


def _verdict(d_liking: float, d_distance: float,
             tol_liking: float = 0.1, tol_distance: float = 0.1) -> str:
    """Heuristic : Liking gain or distance shrink (above tolerance) → improved.
    Both regress → regressed. Otherwise unchanged."""
    improved = (d_liking >  tol_liking) or (d_distance < -tol_distance)
    regressed = (d_liking < -tol_liking) and (d_distance >  tol_distance)
    if regressed: return "regressed"
    if improved:  return "improved"
    return "unchanged"


def _match_weak_segments(
    old: List[WindowSegment], new: List[WindowSegment],
    tol_s: float = 1.5,
) -> List[Dict]:
    """Pair weak segments across reports by time-overlap (centre distance
    within `tol_s`). Returns one dict per match with the Liking delta."""
    pairs: List[Dict] = []
    used_new = set()
    for o in old:
        oc = (o.start_s + o.end_s) / 2
        best_j, best_dt = None, None
        for j, n in enumerate(new):
            if j in used_new:
                continue
            nc = (n.start_s + n.end_s) / 2
            dt = abs(nc - oc)
            if dt <= tol_s and (best_dt is None or dt < best_dt):
                best_j, best_dt = j, dt
        if best_j is None:
            pairs.append({
                "segment": f"{o.start_s:.1f}-{o.end_s:.1f}s",
                "old_Liking": o.Liking_pred, "new_Liking": None,
                "d_Liking": None, "matched": False,
            })
            continue
        n = new[best_j]
        used_new.add(best_j)
        pairs.append({
            "segment": f"{o.start_s:.1f}-{o.end_s:.1f}s",
            "old_Liking": o.Liking_pred, "new_Liking": n.Liking_pred,
            "d_Liking": n.Liking_pred - o.Liking_pred, "matched": True,
        })
    return pairs


# ---------------------------------------------------------------- main
def compare_gaps(old: GapReport, new: GapReport) -> ImprovementSummary:
    d_V = new.V_hat - old.V_hat
    d_A = new.A_hat - old.A_hat
    d_L = new.Liking_pred - old.Liking_pred
    d_D = new.distance - old.distance
    trans = _transition(old.in_zone, new.in_zone)
    verd  = _verdict(d_L, d_D)
    weak  = _match_weak_segments(old.weak, new.weak) if (old.weak or new.weak) else []
    return ImprovementSummary(
        d_V_hat=d_V, d_A_hat=d_A, d_Liking_pred=d_L, d_distance=d_D,
        in_zone_before=old.in_zone, in_zone_after=new.in_zone,
        transition=trans, verdict=verd, weak_changes=weak,
    )


def format_summary(s: ImprovementSummary) -> str:
    """Human-readable diff. The arrow direction follows new − old."""
    arrow = {"improved": "↑", "unchanged": "→", "regressed": "↓"}[s.verdict]
    lines = [
        f"CLOSED-LOOP COMPARISON   verdict: {s.verdict} {arrow}",
        f"  V_hat  : {s.d_V_hat:+.2f}",
        f"  A_hat  : {s.d_A_hat:+.2f}",
        f"  Liking : {s.d_Liking_pred:+.2f}",
        f"  Dist.  : {s.d_distance:+.2f}   (negative = closer to optimal zone)",
        f"  Zone   : {s.in_zone_before} -> {s.in_zone_after}   ({s.transition})",
    ]
    if s.weak_changes:
        lines.append("  Weak segments:")
        for c in s.weak_changes:
            if c["matched"]:
                lines.append(
                    f"    {c['segment']:>14s} : "
                    f"L {c['old_Liking']:.2f} -> {c['new_Liking']:.2f} "
                    f"({c['d_Liking']:+.2f})")
            else:
                lines.append(
                    f"    {c['segment']:>14s} : "
                    f"L {c['old_Liking']:.2f} -> (no match in new report)")
    return "\n".join(lines)
