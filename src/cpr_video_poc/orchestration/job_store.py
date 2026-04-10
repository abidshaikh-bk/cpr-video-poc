from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
import socket
from typing import Any
from uuid import uuid4

from cpr_video_poc.settings import Settings
from cpr_video_poc.utils.yaml_io import dump_yaml


def utc_now() -> datetime:
    return datetime.now(UTC)


def utc_now_iso() -> str:
    return utc_now().isoformat().replace("+00:00", "Z")


def parse_utc_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def make_job_id() -> str:
    return f"{utc_now().strftime('%Y%m%dT%H%M%SZ')}_{uuid4().hex[:8]}"


def default_worker_id() -> str:
    return f"{socket.gethostname()}-{uuid4().hex[:6]}"


@dataclass(slots=True)
class JobClaim:
    job_id: str
    job_dir: Path
    manifest: dict[str, Any]
    status: dict[str, Any]


class JobStore:
    def __init__(self, shared_root: str | Path):
        self.shared_root = Path(shared_root).expanduser().resolve()
        self.jobs_root = self.shared_root / "jobs"
        self.runs_root = self.shared_root / "runs"
        self.ensure_layout()

    def ensure_layout(self) -> None:
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        self.runs_root.mkdir(parents=True, exist_ok=True)

    def create_job(
        self,
        *,
        prompt: str,
        seed: int | None,
        settings: Settings,
        requested_by: str = "cli",
    ) -> dict[str, Any]:
        job_id = make_job_id()
        job_dir = self.job_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=False)

        project_snapshot_path = job_dir / "project_config.yaml"
        backend_snapshot_path = job_dir / "backend_config.yaml"
        dump_yaml(project_snapshot_path, deepcopy(settings.project))
        dump_yaml(backend_snapshot_path, deepcopy(settings.backend))

        manifest = {
            "job_id": job_id,
            "prompt": prompt,
            "seed": seed,
            "submitted_at": utc_now_iso(),
            "requested_by": requested_by,
            "config_files": {
                "project": project_snapshot_path.name,
                "backend": backend_snapshot_path.name,
            },
        }
        status = {
            "job_id": job_id,
            "state": "queued",
            "submitted_at": manifest["submitted_at"],
            "updated_at": manifest["submitted_at"],
            "attempt": 0,
            "worker_id": None,
            "heartbeat_at": None,
            "run_dir": None,
            "error": None,
        }
        self.write_json(job_dir / "job.json", manifest)
        self.write_json(job_dir / "status.json", status)
        return {
            "job_id": job_id,
            "job_dir": job_dir,
            "manifest": manifest,
            "status": status,
        }

    def job_dir(self, job_id: str) -> Path:
        return self.jobs_root / job_id

    def read_manifest(self, job_id: str) -> dict[str, Any]:
        return self.read_json(self.job_dir(job_id) / "job.json")

    def read_status(self, job_id: str) -> dict[str, Any]:
        return self.read_json(self.job_dir(job_id) / "status.json")

    def write_status(self, job_id: str, data: dict[str, Any]) -> None:
        self.write_json(self.job_dir(job_id) / "status.json", data)

    def list_jobs(self) -> list[dict[str, Any]]:
        jobs: list[dict[str, Any]] = []
        for job_file in sorted(self.jobs_root.glob("*/job.json")):
            job_id = job_file.parent.name
            manifest = self.read_json(job_file)
            status = self.read_status(job_id)
            jobs.append(
                {
                    "job_id": job_id,
                    "job_dir": str(job_file.parent),
                    "manifest": manifest,
                    "status": status,
                }
            )
        jobs.sort(key=lambda item: item["manifest"].get("submitted_at", ""))
        return jobs

    def claim_next_job(
        self,
        *,
        worker_id: str,
        stale_after_seconds: int,
    ) -> JobClaim | None:
        for item in self.list_jobs():
            job_id = item["job_id"]
            status = item["status"]
            state = status.get("state")
            if state == "queued":
                if self.acquire_lease(job_id, worker_id, stale_after_seconds):
                    return JobClaim(
                        job_id=job_id,
                        job_dir=self.job_dir(job_id),
                        manifest=self.read_manifest(job_id),
                        status=self.read_status(job_id),
                    )
            elif state == "running" and self.is_stale(job_id, stale_after_seconds):
                self.requeue_job(job_id, reason="stale-worker-reclaim")
                if self.acquire_lease(job_id, worker_id, stale_after_seconds):
                    return JobClaim(
                        job_id=job_id,
                        job_dir=self.job_dir(job_id),
                        manifest=self.read_manifest(job_id),
                        status=self.read_status(job_id),
                    )
        return None

    def acquire_lease(
        self,
        job_id: str,
        worker_id: str,
        stale_after_seconds: int,
    ) -> bool:
        lease_path = self.job_dir(job_id) / "lease.json"
        payload = {
            "job_id": job_id,
            "worker_id": worker_id,
            "claimed_at": utc_now_iso(),
            "heartbeat_at": utc_now_iso(),
        }
        try:
            self.write_json(lease_path, payload, exclusive=True)
        except FileExistsError:
            if not self.is_stale(job_id, stale_after_seconds):
                return False
            lease_path.unlink(missing_ok=True)
            self.write_json(lease_path, payload, exclusive=True)

        status = self.read_status(job_id)
        now = utc_now_iso()
        status.update(
            {
                "state": "running",
                "updated_at": now,
                "worker_id": worker_id,
                "heartbeat_at": now,
                "started_at": status.get("started_at") or now,
                "attempt": int(status.get("attempt", 0)) + 1,
                "error": None,
            }
        )
        self.write_status(job_id, status)
        return True

    def heartbeat(self, job_id: str, worker_id: str) -> None:
        lease_path = self.job_dir(job_id) / "lease.json"
        if not lease_path.exists():
            return
        lease = self.read_json(lease_path)
        if lease.get("worker_id") != worker_id:
            return
        now = utc_now_iso()
        lease["heartbeat_at"] = now
        self.write_json(lease_path, lease)
        status = self.read_status(job_id)
        status["heartbeat_at"] = now
        status["updated_at"] = now
        self.write_status(job_id, status)

    def release_lease(self, job_id: str, worker_id: str) -> None:
        lease_path = self.job_dir(job_id) / "lease.json"
        if not lease_path.exists():
            return
        lease = self.read_json(lease_path)
        if lease.get("worker_id") == worker_id:
            lease_path.unlink(missing_ok=True)

    def is_stale(self, job_id: str, stale_after_seconds: int) -> bool:
        status = self.read_status(job_id)
        heartbeat = parse_utc_timestamp(status.get("heartbeat_at"))
        if heartbeat is None:
            lease_path = self.job_dir(job_id) / "lease.json"
            if not lease_path.exists():
                return True
            lease = self.read_json(lease_path)
            heartbeat = parse_utc_timestamp(lease.get("heartbeat_at"))
            if heartbeat is None:
                return True
        return heartbeat < utc_now() - timedelta(seconds=stale_after_seconds)

    def requeue_job(self, job_id: str, reason: str = "manual-requeue") -> dict[str, Any]:
        status = self.read_status(job_id)
        now = utc_now_iso()
        status.update(
            {
                "state": "queued",
                "updated_at": now,
                "worker_id": None,
                "heartbeat_at": None,
                "error": None,
                "requeue_reason": reason,
            }
        )
        self.write_status(job_id, status)
        (self.job_dir(job_id) / "lease.json").unlink(missing_ok=True)
        return status

    def mark_completed(
        self,
        *,
        job_id: str,
        worker_id: str,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        now = utc_now_iso()
        status = self.read_status(job_id)
        status.update(
            {
                "state": "completed",
                "updated_at": now,
                "completed_at": now,
                "worker_id": worker_id,
                "heartbeat_at": now,
                "run_dir": result.get("run_dir"),
                "video_path": result.get("video_path"),
                "preview_gif_path": result.get("preview_gif_path"),
                "error": None,
            }
        )
        self.write_status(job_id, status)
        self.write_json(self.job_dir(job_id) / "result.json", result)
        return status

    def mark_failed(
        self,
        *,
        job_id: str,
        worker_id: str,
        error: str,
    ) -> dict[str, Any]:
        now = utc_now_iso()
        status = self.read_status(job_id)
        status.update(
            {
                "state": "failed",
                "updated_at": now,
                "failed_at": now,
                "worker_id": worker_id,
                "heartbeat_at": now,
                "error": error,
            }
        )
        self.write_status(job_id, status)
        (self.job_dir(job_id) / "error.txt").write_text(error, encoding="utf-8")
        return status

    def read_json(self, path: str | Path) -> dict[str, Any]:
        with Path(path).open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Expected JSON object in {path}")
        return data

    def write_json(
        self,
        path: str | Path,
        data: dict[str, Any],
        *,
        exclusive: bool = False,
    ) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(data, indent=2, ensure_ascii=False)
        if exclusive:
            with target.open("x", encoding="utf-8") as f:
                f.write(text)
            return
        tmp_path = target.with_suffix(target.suffix + ".tmp")
        tmp_path.write_text(text, encoding="utf-8")
        tmp_path.replace(target)
