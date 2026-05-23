"""Stage 10 — Prompt builder.

Takes:
  - a GapReport from stage 9 (numeric measurements vs optimal zone)
  - AdMetadata supplied by the user (the LLM cannot see the video)
Returns:
  - a (system_prompt, user_prompt) pair ready for Ollama /api/chat
The system prompt pins the role + the JSON schema the LLM must respect.
The user prompt embeds the numeric report + ad metadata.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import List, Optional, Tuple

from ..gap_analysis import GapReport, format_report


# ---------------------------------------------------------------- ad metadata
@dataclass
class AdSegment:
    """Optional structured time segment of the ad (start/end + short label)."""
    start_s: float
    end_s:   float
    label:   str = ""


@dataclass
class AdMetadata:
    """Everything the LLM knows about the ad — provided externally."""
    title:         str = ""
    duration_s:    float = 30.0
    genre:         str = ""        # e.g. "automotive", "luxury", "fast food"
    target_audience: str = ""      # e.g. "18-35 urban, tech-savvy"
    tempo:         str = ""        # e.g. "fast (140 BPM)", "slow ballad"
    palette:       str = ""        # e.g. "warm reds + neon blue at night"
    voiceover:     str = ""        # e.g. "male deep voice", "none"
    segments:      List[AdSegment] = field(default_factory=list)
    free_text:     str = ""        # any extra info the user wants to add

    def to_text(self) -> str:
        lines = []
        if self.title:           lines.append(f"Title           : {self.title}")
        if self.duration_s:      lines.append(f"Duration        : {self.duration_s:g} s")
        if self.genre:           lines.append(f"Genre           : {self.genre}")
        if self.target_audience: lines.append(f"Target audience : {self.target_audience}")
        if self.tempo:           lines.append(f"Tempo           : {self.tempo}")
        if self.palette:         lines.append(f"Palette / look  : {self.palette}")
        if self.voiceover:       lines.append(f"Voice-over      : {self.voiceover}")
        if self.segments:
            lines.append("Segments :")
            for s in self.segments:
                lines.append(f"  {s.start_s:5.2f}s - {s.end_s:5.2f}s : {s.label}")
        if self.free_text:
            lines.append("Notes :")
            lines.append(f"  {self.free_text}")
        return "\n".join(lines) if lines else "(no metadata provided)"


# ---------------------------------------------------------------- prompts
SYSTEM_PROMPT = """You are a senior creative director for video advertising, specialised in
emotion-aware optimisation. You receive:
  1. A NUMERIC EMOTIONAL REPORT measured on a real viewer (Valence, Arousal,
     Liking) compared to the optimal (V*, A*) zone that maximises Liking.
  2. A TEXT DESCRIPTION of the ad. You CANNOT see the video itself.

Your job: propose concrete, actionable creative changes to shift the viewer's
emotional response toward the optimal zone, segment by segment when possible.

Constraints:
- Be specific (timestamps, sensory levers: pacing, cuts, music, palette,
  voice, framing). Avoid generic platitudes.
- Map every recommendation to the numeric gap (which V/A direction it moves
  and why it should raise Liking).
- If a weak segment is listed, address it explicitly.

OUTPUT FORMAT (strict JSON, no prose outside the JSON object):
{
  "overall_assessment": "<2-3 sentence diagnosis of the ad's emotional curve>",
  "global_strategy": "<1 sentence on the overall direction to take>",
  "recommendations": [
    {
      "scope": "segment" | "global",
      "segment_start_s": <float or null>,
      "segment_end_s":   <float or null>,
      "category": "pacing" | "cut" | "music" | "voice" | "palette" |
                   "framing" | "narrative" | "other",
      "action": "<imperative, concrete instruction>",
      "rationale": "<why this moves V or A toward (V*, A*) and raises Liking>",
      "expected_effect": {"dV": <float>, "dA": <float>},
      "priority": 1 | 2 | 3
    }
  ]
}

Rules: 3 to 6 recommendations. priority 1 = highest. Numbers as JSON numbers,
strings as JSON strings. No trailing commas. No markdown."""


def build_prompt(
    report: GapReport,
    ad_metadata: AdMetadata,
    extra_instructions: Optional[str] = None,
) -> Tuple[str, str]:
    """Return (system, user) message bodies for the Ollama chat API."""
    user_blocks = [
        "NUMERIC EMOTIONAL REPORT",
        format_report(report),
        "",
        "AD METADATA (external context — the LLM cannot see the video)",
        ad_metadata.to_text(),
    ]
    if extra_instructions:
        user_blocks += ["", "EXTRA INSTRUCTIONS", extra_instructions]
    user_blocks += ["", "Respond NOW with the JSON object only."]
    return SYSTEM_PROMPT, "\n".join(user_blocks)


def report_to_dict(report: GapReport) -> dict:
    """Convenience: full numeric payload as a dict (useful for logging /
    sending alongside the prompt)."""
    return report.to_dict()


def metadata_to_dict(meta: AdMetadata) -> dict:
    return asdict(meta)
