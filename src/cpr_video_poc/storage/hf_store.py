from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class HFStore:
    def upload_run(self, folder_path: str | Path, repo_id: str | None, private: bool = True) -> bool:
        if not repo_id:
            logger.info("Skipping HF sync because repo_id is not configured.")
            return False
        try:
            from huggingface_hub import HfApi, create_repo, upload_folder

            create_repo(repo_id=repo_id, private=private, exist_ok=True)
            upload_folder(
                repo_id=repo_id,
                folder_path=str(Path(folder_path)),
                path_in_repo=Path(folder_path).name,
            )
            logger.info("Uploaded run folder to Hugging Face repo: %s", repo_id)
            return True
        except Exception as exc:  # pragma: no cover
            logger.warning("HF upload skipped or failed: %s", exc)
            return False
