from cpr_video_poc.pipeline.prompt_builder import build_prompts



def test_build_prompts_includes_context_and_negative_prompt():
    prompts = build_prompts(
        {
            "action": "CPR",
            "ethnicity": "Asian",
            "scene": "office",
            "responders": 2,
            "style": "realistic",
        }
    )
    assert "Asian adult responders" in prompts["prompt"]
    assert "modern office" in prompts["prompt"]
    assert "incorrect anatomy" in prompts["negative_prompt"]


def test_build_prompts_for_single_responder_omits_unknown_ethnicity():
    prompts = build_prompts(
        {
            "action": "CPR",
            "ethnicity": "unknown",
            "scene": "classroom",
            "responders": 1,
            "style": "realistic training",
        }
    )
    assert "One adult responder" in prompts["prompt"]
    assert "unknown" not in prompts["prompt"].lower()
    assert "first-aid classroom" in prompts["prompt"]
