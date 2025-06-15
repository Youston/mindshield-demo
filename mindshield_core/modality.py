"""Rule-based modality scorer.

Given a user message, output scores for each therapeutic modality.
Later this can be replaced by a fine-tuned classifier; for now we use
keyword heuristics to keep implementation light-weight and dependency-free.
"""
from __future__ import annotations

from typing import Dict
import re

_MODALITY_KEYWORDS = {
    "CBT": [r"thought", r"belief", r"negative thinking", r"cognitive"],
    "Mindfulness": [r"present moment", r"meditat", r"breath", r"grounding"],
    "ACT": [r"values", r"accept", r"defus", r"committed action"],
    "Solution-Focused": [r"goal", r"solution", r"next step", r"strength"],
    "Psychodynamic": [r"childhood", r"unconscious", r"dream", r"attachment"],
}


def score_modalities(text: str) -> Dict[str, float]:
    """Return a modality→score dict in range 0-1.

    Scoring is proportional to keyword hits / total keywords per modality.
    If no keywords hit, returns an empty dict.
    """
    text_lower = text.lower()
    scores: Dict[str, float] = {}
    for modality, kws in _MODALITY_KEYWORDS.items():
        hits = sum(1 for kw in kws if re.search(kw, text_lower))
        if hits:
            scores[modality] = hits / len(kws)
    # normalise to 0-1 max
    if scores:
        max_score = max(scores.values())
        for k in scores:
            scores[k] /= max_score or 1.0
    return scores


def choose_modality(scores: Dict[str, float]) -> str:
    """Pick the modality with highest score; empty string if none."""
    if not scores:
        return ""
    return max(scores.items(), key=lambda x: x[1])[0] 