from __future__ import annotations

from pathlib import Path
import shutil


class LocalStore:
    def ensure_dir(self, path: str | Path) -> Path:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def copy_tree(self, src: str | Path, dst: str | Path) -> Path:
        src_path = Path(src)
        dst_path = Path(dst)
        if dst_path.exists():
            shutil.rmtree(dst_path)
        shutil.copytree(src_path, dst_path)
        return dst_path
