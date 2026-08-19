"""The shared Modal image, including the native search extension build.

`pe-search` is a path dependency in `pyproject.toml`, so `uv_sync()` cannot
resolve it without both the crate source and a Rust toolchain present in the
image. Building the wheel here rather than shipping one is not a preference: a
wheel built on a developer's macOS machine cannot load on a Linux Modal worker,
so the compile has to happen on the target platform. The layer is cached and only
rebuilds when the crate changes.

Two details are load-bearing and were both learned from failed builds:

* `uv_sync()` stages the project at ``/.uv`` and runs ``uv sync --project=/.uv``,
  so a relative path dependency resolves against ``/.uv``. The crate must be
  copied to ``/.uv/rust``, not to the image's working directory.
* The toolchain is installed to an explicit ``CARGO_HOME`` instead of relying on
  rustup's default of ``$HOME/.cargo``, so the location does not depend on which
  user or ``HOME`` the builder happens to use.
"""

from __future__ import annotations

from typing import Final

import modal

PYTHON_VERSION: Final[str] = "3.11"

# Where `uv_sync()` stages pyproject.toml and uv.lock; see modal._image.uv_sync.
UV_ROOT: Final[str] = "/.uv"
CARGO_HOME: Final[str] = "/opt/cargo"
RUSTUP_HOME: Final[str] = "/opt/rustup"

RUST_INSTALL: Final[str] = (
    f"curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs "
    f"| CARGO_HOME={CARGO_HOME} RUSTUP_HOME={RUSTUP_HOME} "
    f"sh -s -- -y --profile minimal --default-toolchain stable --no-modify-path"
)

# Fail in the toolchain layer with an obvious message rather than surfacing a
# missing compiler later as an opaque maturin build error.
RUST_VERIFY: Final[str] = f"{CARGO_HOME}/bin/cargo --version && {CARGO_HOME}/bin/rustc --version"

SYSTEM_PATH: Final[str] = "/usr/local/bin:/usr/local/sbin:/usr/bin:/usr/sbin:/bin:/sbin"


def build_image() -> modal.Image:
    """Return the image shared by training and self-play."""

    return (
        modal.Image.debian_slim(python_version=PYTHON_VERSION)
        .apt_install("curl", "build-essential")
        .run_commands(RUST_INSTALL, RUST_VERIFY)
        .env(
            {
                "CARGO_HOME": CARGO_HOME,
                "RUSTUP_HOME": RUSTUP_HOME,
                "PATH": f"{CARGO_HOME}/bin:{SYSTEM_PATH}",
            }
        )
        # uv resolves the `rust/pe-search` path dependency relative to UV_ROOT,
        # and needs it on disk before `uv sync` runs.
        .add_local_dir("rust", f"{UV_ROOT}/rust", copy=True)
        .uv_sync()
        .add_local_python_source("pink_elephant")
    )
