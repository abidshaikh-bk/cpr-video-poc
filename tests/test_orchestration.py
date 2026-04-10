from datetime import timedelta
from pathlib import Path

from cpr_video_poc.orchestration.job_store import JobStore, utc_now_iso
from cpr_video_poc.orchestration.worker import process_job
from cpr_video_poc.settings import load_settings


def test_create_job_writes_manifest_status_and_snapshots(tmp_path: Path):
    settings = load_settings()
    store = JobStore(tmp_path / "shared")

    created = store.create_job(
        prompt="Generate a CPR video",
        seed=42,
        settings=settings,
    )

    job_dir = created["job_dir"]
    assert (job_dir / "job.json").exists()
    assert (job_dir / "status.json").exists()
    assert (job_dir / "project_config.yaml").exists()
    assert (job_dir / "backend_config.yaml").exists()
    assert created["status"]["state"] == "queued"


def test_claim_next_job_reclaims_stale_running_job(tmp_path: Path):
    settings = load_settings()
    store = JobStore(tmp_path / "shared")
    created = store.create_job(
        prompt="Generate a CPR video",
        seed=42,
        settings=settings,
    )
    job_id = created["job_id"]

    old_timestamp = (store.read_status(job_id)["submitted_at"]).replace("Z", "+00:00")
    stale_heartbeat = utc_now_iso()
    status = store.read_status(job_id)
    status.update(
        {
            "state": "running",
            "worker_id": "dead-worker",
            "heartbeat_at": "2000-01-01T00:00:00Z",
        }
    )
    store.write_status(job_id, status)
    store.write_json(
        store.job_dir(job_id) / "lease.json",
        {
            "job_id": job_id,
            "worker_id": "dead-worker",
            "claimed_at": old_timestamp,
            "heartbeat_at": "2000-01-01T00:00:00Z",
        },
    )

    claim = store.claim_next_job(worker_id="worker-2", stale_after_seconds=1)

    assert claim is not None
    assert claim.job_id == job_id
    assert store.read_status(job_id)["state"] == "running"
    assert store.read_status(job_id)["worker_id"] == "worker-2"


def test_process_job_marks_completed_with_fake_generation(tmp_path: Path, monkeypatch):
    settings = load_settings()
    store = JobStore(tmp_path / "shared")
    created = store.create_job(
        prompt="Generate a CPR video",
        seed=123,
        settings=settings,
    )
    claim = store.claim_next_job(worker_id="worker-1", stale_after_seconds=60)
    assert claim is not None

    def fake_run_generation_with_settings(*args, **kwargs):
        return {
            "run_dir": str(store.runs_root / "fake-run"),
            "video_path": str(store.runs_root / "fake-run" / "output.mp4"),
            "preview_gif_path": str(store.runs_root / "fake-run" / "preview.gif"),
            "seed": 123,
            "parsed_prompt": {"action": "CPR"},
            "final_prompt": "final",
            "negative_prompt": "negative",
            "synced_to_hf": False,
        }

    monkeypatch.setattr(
        "cpr_video_poc.orchestration.worker.run_generation_with_settings",
        fake_run_generation_with_settings,
    )

    result = process_job(
        store=store,
        claim=claim,
        worker_id="worker-1",
        heartbeat_interval_seconds=1,
    )

    status = store.read_status(created["job_id"])
    assert result["state"] == "completed"
    assert status["state"] == "completed"
    assert Path(store.job_dir(created["job_id"]) / "result.json").exists()
