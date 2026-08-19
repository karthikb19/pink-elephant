"""The shared Modal image, including the native search extension build.

`pe-search` is a path dependency in `pyproject.toml`, so `uv_sync()` cannot
resolve it without both the crate source and a Rust toolchain present in the
image. Building the wheel here rather than shipping one is not a preference: a
wheel built on a developer's macOS machine cannot load on a Linux Modal worker,
so the compile has to happen on the target platform. The layer is cached and only
rebuilds when the crate changes.
"""

from __future__ import annotations

from typing import Final

import modal

PYTHON_VERSION: Final[str] = "3.11"
RUST_INSTALL: Final[str] = (
    "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs "
    "| sh -s -- -y --profile minimal --default-toolchain stable"
)


def build_image() -> modal.Image:
    """Return the image shared by training and self-play."""

    return (
        modal.Image.debian_slim(python_version=PYTHON_VERSION)
        .apt_install("curl", "build-essential")
        .run_commands(RUST_INSTALL)
        .env({"PATH": "/root/.cargo/bin:/usr/local/bin:/usr/bin:/bin"})
        # The crate must be present before uv resolves the path dependency.
        .add_local_dir("rust", "/root/rust", copy=True)
        .uv_sync()
        .add_local_python_source("pink_elephant")
    )
