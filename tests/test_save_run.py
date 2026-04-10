from pathlib import Path

from PIL import Image

from cpr_video_poc.pipeline.save_run import save_run_artifacts



def test_save_run_artifacts_creates_expected_files(tmp_path: Path):
    frames = [Image.new("RGB", (64, 64), color="white") for _ in range(2)]
    saved = save_run_artifacts(
        run_root=tmp_path,
        request_data={"prompt": "test"},
        parsed_prompt={"action": "CPR"},
        final_prompt="final prompt",
        negative_prompt="negative prompt",
        config={"project": {}},
        metadata={"seed": 1},
        frames=frames,
        fps=4,
    )
    run_dir = saved["run_dir"]
    assert (run_dir / "request.json").exists()
    assert (run_dir / "parsed_prompt.json").exists()
    assert (run_dir / "final_prompt.txt").exists()
    assert (run_dir / "negative_prompt.txt").exists()
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "metadata.json").exists()
    assert (run_dir / "output.mp4").exists()
    assert (run_dir / "preview.gif").exists()
