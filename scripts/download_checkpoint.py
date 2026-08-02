"""Download one checkpoint from the project's Modal Volume."""

from __future__ import annotations

import argparse
import subprocess
from collections.abc import Sequence
from pathlib import Path, PurePosixPath

DEFAULT_VOLUME_NAME = "pink-elephant-training"


def checkpoint_remote_path(run_name: str, checkpoint: str) -> str:
    """Build and validate the remote path used by Modal Volume get."""

    safe_run_name = _safe_relative_path(run_name, label="run name")
    safe_checkpoint = _safe_checkpoint_name(checkpoint)
    return str(PurePosixPath("runs") / safe_run_name / safe_checkpoint)


def download_checkpoint(
    run_name: str,
    checkpoint: str,
    *,
    output_dir: Path,
    volume_name: str = DEFAULT_VOLUME_NAME,
) -> Path:
    """Download a checkpoint and return its expected local path."""

    remote_path = checkpoint_remote_path(run_name, checkpoint)
    checkpoint_name = _safe_checkpoint_name(checkpoint)
    output_dir = output_dir.expanduser().resolve()
    output_path = output_dir / checkpoint_name
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing checkpoint: {output_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["uv", "run", "modal", "volume", "get", volume_name, remote_path, str(output_dir)],
        check=True,
    )
    if not output_path.is_file():
        raise FileNotFoundError(
            f"Modal completed but the expected checkpoint was not found: {output_path}"
        )
    return output_path


def main(argv: Sequence[str] | None = None) -> None:
    """Parse download arguments and retrieve one immutable checkpoint."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--volume-name", default=DEFAULT_VOLUME_NAME)
    args = parser.parse_args(argv)
    local_path = download_checkpoint(
        args.run_name,
        args.checkpoint,
        output_dir=args.output_dir,
        volume_name=args.volume_name,
    )
    print(f"Downloaded {local_path}")


def _safe_relative_path(value: str, *, label: str) -> PurePosixPath:
    """Reject absolute paths and parent traversal before building a remote path."""

    path = PurePosixPath(value)
    if not value or path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"{label} must be a non-empty relative path: {value!r}")
    return path


def _safe_checkpoint_name(value: str) -> str:
    """Require a single checkpoint filename rather than a remote subpath."""

    path = _safe_relative_path(value, label="checkpoint")
    if len(path.parts) != 1 or path.suffix != ".pt":
        raise ValueError("checkpoint must be a single .pt filename")
    return path.name


if __name__ == "__main__":
    main()
