from __future__ import annotations

import re
from typing import Any


ETHNICITY_PATTERNS = {
    "Asian": re.compile(r"\b(asian|east asian|south asian|southeast asian)\b", re.IGNORECASE),
    "African": re.compile(r"\b(african|black)\b", re.IGNORECASE),
}

SCENE_PATTERNS = {
    "office": re.compile(r"\boffice\b", re.IGNORECASE),
    "classroom": re.compile(r"\bclassroom\b", re.IGNORECASE),
    "hospital": re.compile(r"\bhospital\b", re.IGNORECASE),
}

ACTION_PATTERNS = (
    re.compile(r"\bcpr\b", re.IGNORECASE),
    re.compile(r"\bcardiopulmonary resuscitation\b", re.IGNORECASE),
    re.compile(r"\bchest compressions\b", re.IGNORECASE),
)

ONE_RESPONDER_PATTERN = re.compile(
    r"\b(one responder|single responder|one person|solo responder|lone responder)\b",
    re.IGNORECASE,
)
TWO_RESPONDER_PATTERN = re.compile(
    r"\b(two responders|two people|two-person|pair of responders|two adults)\b",
    re.IGNORECASE,
)


def _detect_action(prompt: str) -> str:
    return "CPR" if any(pattern.search(prompt) for pattern in ACTION_PATTERNS) else "unknown"


def _detect_ethnicity(prompt: str) -> str:
    for label, pattern in ETHNICITY_PATTERNS.items():
        if pattern.search(prompt):
            return label
    return "unknown"


def _detect_scene(prompt: str) -> str:
    for label, pattern in SCENE_PATTERNS.items():
        if pattern.search(prompt):
            return label
    if re.search(r"\b(workplace|workplace safety)\b", prompt, re.IGNORECASE):
        return "office"
    return "unknown"


def _detect_responders(prompt: str) -> int:
    if ONE_RESPONDER_PATTERN.search(prompt):
        return 1
    if TWO_RESPONDER_PATTERN.search(prompt):
        return 2
    return 2


def _detect_style(prompt: str) -> str:
    if re.search(r"\b(training|safety|instructional)\b", prompt, re.IGNORECASE):
        return "realistic training"
    return "realistic"


def parse_prompt(prompt: str) -> dict[str, Any]:
    text = " ".join(prompt.strip().split())
    return {
        "action": _detect_action(text),
        "ethnicity": _detect_ethnicity(text),
        "scene": _detect_scene(text),
        "responders": _detect_responders(text),
        "style": _detect_style(text),
        "raw_prompt": text,
    }
