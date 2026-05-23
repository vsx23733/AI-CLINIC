"""Advisor sub-package — stages 10 to 13.

Boundary note: from stage 10 onwards, the actual ad cannot come from DEAP
(DEAP has no video). The user supplies an AdMetadata description; the LLM
turns the numeric gap + that description into creative recommendations.
"""
__all__ = [
    "prompt_builder",
    "llm_advisor",
    "recommendations",
    "closed_loop",
]
