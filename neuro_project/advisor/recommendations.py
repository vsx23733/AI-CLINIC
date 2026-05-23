"""Stage 12 — parse, validate, and export the LLM's structured advice.

The LLM is asked (in prompt_builder) to return strict JSON. This module:
  - tolerates minor noise (extra prose, code-fences) with a JSON extractor
  - validates the schema (coerces types, drops malformed entries)
  - exposes a normalised RecommendationSet for downstream code / UI
  - renders a human-readable text view
  - exports to JSON file
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

VALID_SCOPES = {"segment", "global"}
VALID_CATEGORIES = {"pacing", "cut", "music", "voice", "palette",
                    "framing", "narrative", "other"}


# ---------------------------------------------------------------- dataclasses
@dataclass
class Recommendation:
    scope:            str
    category:         str
    action:           str
    rationale:        str
    segment_start_s:  Optional[float] = None
    segment_end_s:    Optional[float] = None
    expected_dV:      Optional[float] = None
    expected_dA:      Optional[float] = None
    priority:         int = 2

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RecommendationSet:
    overall_assessment: str = ""
    global_strategy:    str = ""
    recommendations:    List[Recommendation] = field(default_factory=list)
    parse_warnings:     List[str] = field(default_factory=list)
    raw_text:           str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_assessment": self.overall_assessment,
            "global_strategy":    self.global_strategy,
            "recommendations":    [r.to_dict() for r in self.recommendations],
            "parse_warnings":     self.parse_warnings,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


# ---------------------------------------------------------------- JSON extraction
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Try multiple strategies to locate a JSON object inside `text`.
    Returns the parsed dict or None."""
    if not text:
        return None
    candidates: List[str] = []
    # 1. straight parse
    candidates.append(text.strip())
    # 2. fenced ```json ... ``` block
    for m in _FENCE_RE.findall(text):
        candidates.append(m.strip())
    # 3. first '{' ... matching '}' span (greedy)
    first = text.find("{")
    last  = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        candidates.append(text[first:last + 1])

    for c in candidates:
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


# ---------------------------------------------------------------- coercion
def _coerce_float(v: Any) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _coerce_int(v: Any, default: int = 2) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _coerce_str(v: Any) -> str:
    return "" if v is None else str(v).strip()


# ---------------------------------------------------------------- main parser
def parse_advice(text: str) -> RecommendationSet:
    """Turn LLM raw output into a validated RecommendationSet. Never raises ;
    issues are accumulated in `parse_warnings` for visibility."""
    rs = RecommendationSet(raw_text=text or "")
    obj = _extract_json_object(text)
    if obj is None:
        rs.parse_warnings.append("Could not extract a JSON object from the LLM output.")
        return rs

    rs.overall_assessment = _coerce_str(obj.get("overall_assessment"))
    rs.global_strategy    = _coerce_str(obj.get("global_strategy"))

    raw_list = obj.get("recommendations")
    if not isinstance(raw_list, list):
        rs.parse_warnings.append("Field 'recommendations' is missing or not a list.")
        return rs

    for i, item in enumerate(raw_list):
        if not isinstance(item, dict):
            rs.parse_warnings.append(f"recommendation[{i}] is not a dict, dropped.")
            continue
        scope    = _coerce_str(item.get("scope")).lower()
        category = _coerce_str(item.get("category")).lower()
        action   = _coerce_str(item.get("action"))
        rationale = _coerce_str(item.get("rationale"))

        if not action:
            rs.parse_warnings.append(f"recommendation[{i}] has no action, dropped.")
            continue
        if scope not in VALID_SCOPES:
            rs.parse_warnings.append(
                f"recommendation[{i}] scope='{scope}' invalid, coerced to 'global'.")
            scope = "global"
        if category not in VALID_CATEGORIES:
            rs.parse_warnings.append(
                f"recommendation[{i}] category='{category}' invalid, coerced to 'other'.")
            category = "other"

        exp = item.get("expected_effect") or {}
        if not isinstance(exp, dict):
            exp = {}

        rec = Recommendation(
            scope=scope, category=category,
            action=action, rationale=rationale,
            segment_start_s=_coerce_float(item.get("segment_start_s")),
            segment_end_s  =_coerce_float(item.get("segment_end_s")),
            expected_dV    =_coerce_float(exp.get("dV")),
            expected_dA    =_coerce_float(exp.get("dA")),
            priority       =_coerce_int(item.get("priority"), default=2),
        )
        rs.recommendations.append(rec)

    rs.recommendations.sort(key=lambda r: (r.priority, r.scope != "global"))
    return rs


# ---------------------------------------------------------------- exports
def format_recommendations(rs: RecommendationSet) -> str:
    """Human-readable rendering (for terminal / report)."""
    lines = []
    if rs.overall_assessment:
        lines += ["OVERALL ASSESSMENT", "  " + rs.overall_assessment, ""]
    if rs.global_strategy:
        lines += ["GLOBAL STRATEGY",    "  " + rs.global_strategy, ""]
    if not rs.recommendations:
        lines.append("(no recommendations)")
    else:
        lines.append("RECOMMENDATIONS")
        for i, r in enumerate(rs.recommendations, 1):
            scope_str = (f"{r.segment_start_s:.1f}s-{r.segment_end_s:.1f}s"
                         if r.scope == "segment" and r.segment_start_s is not None
                         else "global")
            dV = "" if r.expected_dV is None else f" dV={r.expected_dV:+.2f}"
            dA = "" if r.expected_dA is None else f" dA={r.expected_dA:+.2f}"
            lines.append(
                f"  [{r.priority}] {scope_str:>14s} | {r.category:9s} | "
                f"{r.action}")
            if r.rationale:
                lines.append(f"      ↳ {r.rationale}{dV}{dA}")
    if rs.parse_warnings:
        lines += ["", "PARSE WARNINGS"]
        lines += [f"  - {w}" for w in rs.parse_warnings]
    return "\n".join(lines)


def save_recommendations(
    rs: RecommendationSet, path: str, also_text: bool = True,
) -> str:
    """Write JSON to `path` (.json). If `also_text`, write a sibling .txt with
    the human-readable view. Returns the JSON path."""
    base, ext = os.path.splitext(path)
    if ext.lower() != ".json":
        path = base + ".json"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(rs.to_json())
    if also_text:
        with open(base + ".txt", "w", encoding="utf-8") as f:
            f.write(format_recommendations(rs))
    return path
