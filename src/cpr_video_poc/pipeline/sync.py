from __future__ import annotations

from pathlib import Path

from cpr_video_poc.storage.hf_store import HFStore



def sync_run_if_enabled(run_dir: str | Path, storage_cfg: dict) -> bool:
    if not bool(storage_cfg.get("sync_to_hf", False)):
        return False
    repo_id = storage_cfg.get("hf_repo_id")
    private = bool(storage_cfg.get("hf_private", True))
    return HFStore().upload_run(run_dir, repo_id=repo_id, private=private)
