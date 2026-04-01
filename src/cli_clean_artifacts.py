from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def _safe_clear(path: Path, *, assume_yes: bool = False) -> None:
    path = path.expanduser().resolve()
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        return
    if not path.is_dir():
        raise SystemExit(f"Refusing to clear non-directory path: {path}")
    if path == path.parent:
        raise SystemExit(f"Refusing to clear root-like path: {path}")
    contents = list(path.iterdir())
    if contents:
        print(f"WARNING: about to delete {len(contents)} items under {path}")
        if not assume_yes:
            answer = input("  Type 'yes' to confirm: ").strip().lower()
            if answer != "yes":
                raise SystemExit("Aborted by user.")
    shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Clear generated experiment artifacts for the paper repo.")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--experiments-root", required=True)
    parser.add_argument("--logs-root", required=True)
    parser.add_argument("--yes", action="store_true", help="Skip interactive confirmation")
    args = parser.parse_args()

    _safe_clear(Path(args.out_root), assume_yes=args.yes)
    _safe_clear(Path(args.experiments_root), assume_yes=args.yes)
    _safe_clear(Path(args.logs_root), assume_yes=args.yes)


if __name__ == "__main__":
    main()
