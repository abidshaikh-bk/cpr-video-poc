from cpr_video_poc.pipeline.prompt_parser import parse_prompt



def test_parse_prompt_extracts_expected_fields():
    parsed = parse_prompt("Generate a video with Asian people giving CPR in an office")
    assert parsed["action"] == "CPR"
    assert parsed["ethnicity"] == "Asian"
    assert parsed["scene"] == "office"
    assert parsed["responders"] == 2
    assert parsed["style"] == "realistic"


def test_parse_prompt_handles_single_responder_and_workplace_language():
    parsed = parse_prompt("Generate a workplace safety CPR video with one responder")
    assert parsed["action"] == "CPR"
    assert parsed["scene"] == "office"
    assert parsed["responders"] == 1
    assert parsed["style"] == "realistic training"
