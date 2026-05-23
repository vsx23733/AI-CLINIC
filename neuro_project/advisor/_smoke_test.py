"""Smoke-test for the advisor sub-package (stages 10-13).

Runs WITHOUT Ollama : the LLM call is replaced by a fixture JSON. If Ollama is
reachable, it does an additional live ping + real call at the end."""
from __future__ import annotations

import os
import sys
import tempfile

# Force UTF-8 stdout (Windows console defaults to cp1252 and chokes on arrows)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

HERE = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(PKG_ROOT))

from neuro_project.gap_analysis import GapReport, WindowSegment            # noqa: E402
from neuro_project.advisor.prompt_builder import (                          # noqa: E402
    AdMetadata, AdSegment, build_prompt,
)
from neuro_project.advisor.recommendations import (                         # noqa: E402
    parse_advice, format_recommendations, save_recommendations,
)
from neuro_project.advisor.closed_loop import compare_gaps, format_summary  # noqa: E402
from neuro_project.advisor.llm_advisor import OllamaClient, OllamaConfig    # noqa: E402


def section(name: str) -> None:
    print(f"\n=== {name} ===")


# ---------------------------------------------------------------- fixtures
def _make_report(V: float, A: float, L: float, dist: float, in_zone: bool,
                 weak: list[WindowSegment] | None = None) -> GapReport:
    return GapReport(
        V_hat=V, A_hat=A, Liking_pred=L,
        V_star=7.0, A_star=5.5, Liking_star=8.5,
        dV=7.0 - V, dA=5.5 - A, in_zone=in_zone, distance=dist,
        n_windows=200, weak=weak or [],
    )


report_old = _make_report(
    V=4.0, A=5.0, L=4.8, dist=3.04, in_zone=False,
    weak=[
        WindowSegment(28.0, 30.0, 3.1, 4.5, 3.5),
        WindowSegment(34.0, 36.0, 3.4, 4.8, 3.9),
    ],
)
report_new = _make_report(
    V=6.0, A=5.4, L=7.5, dist=1.01, in_zone=True,
    weak=[WindowSegment(28.5, 30.5, 5.5, 5.4, 6.8)],
)

ad = AdMetadata(
    title="EcoBoost Drive 30s",
    duration_s=30.0,
    genre="automotive",
    target_audience="25-45 urban, eco-curious",
    tempo="moderate, builds to fast finale",
    palette="cool blues with a warm sunset payoff",
    voiceover="female warm voice",
    segments=[
        AdSegment(0.0, 8.0, "morning commute, calm"),
        AdSegment(8.0, 22.0, "open road, music swell"),
        AdSegment(22.0, 30.0, "logo + tagline reveal"),
    ],
    free_text="hero shot at 0:18 ; tagline 'Drive forward, lightly'.",
)


# ---------------------------------------------------------------- stage 10
section("Stage 10 — build_prompt")
system, user = build_prompt(report_old, ad)
assert "JSON" in system
assert "Valence" in user and "AD METADATA" in user
print(f"  system prompt : {len(system)} chars")
print(f"  user prompt   : {len(user)} chars (first 200) v")
print("  " + user[:200].replace("\n", "\n  "))


# ---------------------------------------------------------------- stage 12 (parse mock)
section("Stage 12 — parse_advice on a mocked LLM response")
mock_json = """
{
  "overall_assessment": "Emotional curve is flat with a dip mid-roll (28-36s).",
  "global_strategy":   "Lift valence early; spike arousal in the dip.",
  "recommendations": [
    {"scope": "segment", "segment_start_s": 28.0, "segment_end_s": 36.0,
     "category": "pacing",
     "action": "introduce two quick cuts and a music swell",
     "rationale": "weak segment with very low predicted Liking, low arousal",
     "expected_effect": {"dV": 0.4, "dA": 1.2}, "priority": 1},
    {"scope": "global", "category": "palette",
     "action": "warm up colour grading by 10% in mids",
     "rationale": "valence median 4.0 is below V*=7.0",
     "expected_effect": {"dV": 1.0, "dA": 0.0}, "priority": 2},
    {"scope": "segment", "segment_start_s": 22.0, "segment_end_s": 30.0,
     "category": "music",
     "action": "extend the music swell by 2s, delay the cut on logo",
     "rationale": "carry arousal into the logo reveal",
     "expected_effect": {"dV": 0.2, "dA": 0.6}, "priority": 2},
    {"scope": "garbage", "category": "weird", "action": "",
     "rationale": "this entry is intentionally broken to test validation",
     "priority": "high"}
  ]
}
"""
rs = parse_advice(mock_json)
print(f"  parsed       : {len(rs.recommendations)} recos kept")
print(f"  warnings     : {len(rs.parse_warnings)} (expected: 2 — 1 empty action, 1 invalid scope)")
print(format_recommendations(rs))
assert len(rs.recommendations) == 3
assert any("empty" in w.lower() or "no action" in w.lower() for w in rs.parse_warnings) \
       or any("dropped" in w.lower() for w in rs.parse_warnings)


# save + reload round-trip
section("Stage 12 — save_recommendations round-trip")
with tempfile.TemporaryDirectory() as td:
    p = save_recommendations(rs, os.path.join(td, "advice.json"))
    txt = os.path.splitext(p)[0] + ".txt"
    assert os.path.exists(p) and os.path.exists(txt)
    print(f"  wrote {p} ({os.path.getsize(p)} B) and {txt}")


# ---------------------------------------------------------------- stage 13
section("Stage 13 — compare_gaps")
summary = compare_gaps(report_old, report_new)
print(format_summary(summary))
assert summary.verdict == "improved"
assert summary.transition == "entered_zone"
assert summary.weak_changes and summary.weak_changes[0]["matched"]


# ---------------------------------------------------------------- stage 11 (Ollama live, optional)
section("Stage 11 — Ollama live call (optional)")
client = OllamaClient(OllamaConfig(model="gemma4:latest", temperature=0.3))
if not client.ping():
    print("  Ollama unreachable at http://localhost:11434 — skipped.")
    print("  To run the live test : `ollama serve` and `ollama pull llama3.1`.")
else:
    print("  Ollama reachable. Sending real prompt...")
    try:
        from neuro_project.advisor.llm_advisor import advise
        raw = advise(report_old, ad, OllamaConfig(model="gemma4:latest", temperature=0.3))
        print(f"  raw response : {len(raw)} chars (first 200) v")
        print("  " + raw[:200].replace("\n", "\n  "))
        live = parse_advice(raw)
        print(f"  parsed       : {len(live.recommendations)} recommendations")
    except Exception as e:                                        # noqa: BLE001
        print(f"  Live call failed : {e}")


print("\nALL ADVISOR CHECKS PASSED")
