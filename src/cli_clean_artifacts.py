from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def _safe_clear(path: Path) -> None:
    path = path.expanduser().resolve()
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        return
    if not path.is_dir():
        raise SystemExit(f"Refusing to clear non-directory path: {path}")
    if path == path.parent:
        raise SystemExit(f"Refusing to clear root-like path: {path}")
    shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Clear generated experiment artifacts for the paper repo.")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--experiments-root", required=True)
    parser.add_argument("--logs-root", required=True)
    args = parser.parse_args()

    _safe_clear(Path(args.out_root))
    _safe_clear(Path(args.experiments_root))
    _safe_clear(Path(args.logs_root))


if __name__ == "__main__":
    main()
