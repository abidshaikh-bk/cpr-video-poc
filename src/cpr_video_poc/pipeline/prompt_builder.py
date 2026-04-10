from __future__ import annotations

from typing import Any


DEFAULT_NEGATIVE_TERMS = [
    "low quality",
    "blurry",
    "deformed limbs",
    "incorrect anatomy",
    "extra fingers",
    "bad motion",
    "subtitles",
    "text overlays",
]

SCENE_DESCRIPTIONS = {
    "office": "a modern office",
    "classroom": "a practical first-aid classroom",
    "hospital": "a realistic hospital training area",
    "unknown": "an indoor first-aid training space",
}


def build_negative_prompt(extra_terms: list[str] | None = None) -> str:
    terms = list(DEFAULT_NEGATIVE_TERMS)
    if extra_terms:
        terms.extend(extra_terms)
    return ", ".join(dict.fromkeys(terms))


def _build_responder_phrase(ethnicity: str, responders: int) -> str:
    if responders == 1:
        return (
            f"One {ethnicity} adult responder"
            if ethnicity.lower() != "unknown"
            else "One adult responder"
        )
    return (
        f"Two {ethnicity} adult responders"
        if ethnicity.lower() != "unknown"
        else "Two adult responders"
    )


def build_prompts(parsed: dict[str, Any]) -> dict[str, str]:
    ethnicity = str(parsed.get("ethnicity", "unknown"))
    scene = str(parsed.get("scene", "unknown"))
    responders = int(parsed.get("responders", 2))
    style = str(parsed.get("style", "realistic training"))

    responder_phrase = _build_responder_phrase(ethnicity, responders)
    scene_description = SCENE_DESCRIPTIONS.get(scene, SCENE_DESCRIPTIONS["unknown"])

    action_sentence = (
        "One responder performs chest compressions with straight arms and correct hand placement."
    )
    if responders > 1:
        support_sentence = (
            "The second responder stays close to assist, monitor the patient, and support a coordinated training scenario."
        )
    else:
        support_sentence = (
            "The responder maintains focused posture and clear instructional body mechanics."
        )

    prompt = " ".join(
        [
            f"A {style} CPR instruction scene in {scene_description}.",
            f"{responder_phrase} kneel beside an unconscious adult lying supine on the floor.",
            action_sentence,
            support_sentence,
            "Realistic clothing, natural lighting, medium camera shot, clear posture, and a professional training-video composition.",
        ]
    )

    return {
        "prompt": prompt,
        "negative_prompt": build_negative_prompt(),
    }
