from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import threading
import time
import traceback
from typing import Any

from cpr_video_poc.logging_utils import configure_logging
from cpr_video_poc.orchestration.job_store import JobClaim, JobStore, default_worker_id
from cpr_video_poc.pipeline.generate import run_generation_with_settings
from cpr_video_poc.settings import load_settings

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class WorkerConfig:
    shared_root: Path
    worker_id: str
    poll_interval_seconds: int = 15
    heartbeat_interval_seconds: int = 30
    stale_after_seconds: int = 300
    once: bool = False
    max_jobs: int | None = None


def _heartbeat_loop(
    *,
    store: JobStore,
    job_id: str,
    worker_id: str,
    heartbeat_interval_seconds: int,
    stop_event: threading.Event,
) -> None:
    while not stop_event.wait(heartbeat_interval_seconds):
        store.heartbeat(job_id, worker_id)


def process_job(
    *,
    store: JobStore,
    claim: JobClaim,
    worker_id: str,
    heartbeat_interval_seconds: int,
) -> dict[str, Any]:
    job_dir = claim.job_dir
    manifest = claim.manifest
    settings = load_settings(
        job_dir / manifest["config_files"]["project"],
        job_dir / manifest["config_files"]["backend"],
    )

    stop_event = threading.Event()
    heartbeat_thread = threading.Thread(
        target=_heartbeat_loop,
        kwargs={
            "store": store,
            "job_id": claim.job_id,
            "worker_id": worker_id,
            "heartbeat_interval_seconds": heartbeat_interval_seconds,
            "stop_event": stop_event,
        },
        daemon=True,
    )
    heartbeat_thread.start()
    try:
        result = run_generation_with_settings(
            settings,
            prompt=manifest["prompt"],
            seed=manifest.get("seed"),
            artifact_root_override=store.runs_root,
            extra_metadata={
                "job": {
                    "job_id": claim.job_id,
                    "worker_id": worker_id,
                    "shared_root": str(store.shared_root),
                }
            },
        )
        store.mark_completed(job_id=claim.job_id, worker_id=worker_id, result=result)
        logger.info("Completed job %s", claim.job_id)
        return {
            "job_id": claim.job_id,
            "state": "completed",
            "run_dir": result.get("run_dir"),
        }
    except Exception:
        error_text = traceback.format_exc()
        store.mark_failed(job_id=claim.job_id, worker_id=worker_id, error=error_text)
        logger.exception("Job %s failed", claim.job_id)
        return {
            "job_id": claim.job_id,
            "state": "failed",
            "error": error_text,
        }
    finally:
        stop_event.set()
        heartbeat_thread.join(timeout=1)
        store.release_lease(claim.job_id, worker_id)


def run_worker(
    *,
    shared_root: str | Path,
    worker_id: str | None = None,
    poll_interval_seconds: int = 15,
    heartbeat_interval_seconds: int = 30,
    stale_after_seconds: int = 300,
    once: bool = False,
    max_jobs: int | None = None,
) -> dict[str, Any]:
    configure_logging()
    resolved_worker_id = worker_id or default_worker_id()
    store = JobStore(shared_root)
    processed: list[dict[str, Any]] = []

    while True:
        claim = store.claim_next_job(
            worker_id=resolved_worker_id,
            stale_after_seconds=stale_after_seconds,
        )
        if claim is None:
            if once:
                break
            if max_jobs is not None and len(processed) >= max_jobs:
                break
            time.sleep(poll_interval_seconds)
            continue

        processed.append(
            process_job(
                store=store,
                claim=claim,
                worker_id=resolved_worker_id,
                heartbeat_interval_seconds=heartbeat_interval_seconds,
            )
        )
        if once:
            break
        if max_jobs is not None and len(processed) >= max_jobs:
            break

    return {
        "worker_id": resolved_worker_id,
        "processed_jobs": processed,
        "processed_count": len(processed),
        "shared_root": str(Path(shared_root).expanduser().resolve()),
    }
