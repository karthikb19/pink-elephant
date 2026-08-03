"""Download and control a local Stockfish UCI engine."""

from __future__ import annotations

import json
import os
import platform
import shutil
import stat
import tarfile
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TypedDict, cast
from urllib.request import Request, urlopen

import chess
import chess.engine

STOCKFISH_RELEASES_API_URL: Final[str] = (
    "https://api.github.com/repos/official-stockfish/Stockfish/releases/latest"
)
MIN_UCI_ELO: Final[int] = 1320
MAX_UCI_ELO: Final[int] = 3190


class _ReleaseAssetPayload(TypedDict):
    name: str
    browser_download_url: str


class _ReleasePayload(TypedDict):
    assets: list[_ReleaseAssetPayload]


@dataclass(frozen=True, slots=True)
class StockfishAsset:
    """A downloadable platform-specific Stockfish release asset."""

    name: str
    download_url: str


@dataclass(frozen=True, slots=True)
class StockfishConfig:
    """UCI options used for one Stockfish opponent."""

    elo: int = 1400
    depth: int = 10
    movetime_ms: int | None = None
    threads: int = 1
    hash_mb: int = 128

    def __post_init__(self) -> None:
        if not MIN_UCI_ELO <= self.elo <= MAX_UCI_ELO:
            raise ValueError(f"elo must be between {MIN_UCI_ELO} and {MAX_UCI_ELO}, got {self.elo}")
        if self.depth < 1:
            raise ValueError(f"depth must be positive, got {self.depth}")
        if self.movetime_ms is not None and self.movetime_ms < 1:
            raise ValueError(f"movetime_ms must be positive, got {self.movetime_ms}")
        if self.threads < 1:
            raise ValueError(f"threads must be positive, got {self.threads}")
        if self.hash_mb < 1:
            raise ValueError(f"hash_mb must be positive, got {self.hash_mb}")

    def search_limit(self) -> chess.engine.Limit:
        """Return the configured python-chess search limit."""

        if self.movetime_ms is not None:
            return chess.engine.Limit(time=self.movetime_ms / 1000)
        return chess.engine.Limit(depth=self.depth)


@dataclass(slots=True)
class StockfishPlayer:
    """Choose legal moves from a running Stockfish engine."""

    engine: chess.engine.SimpleEngine
    limit: chess.engine.Limit

    def choose_move(self, board: chess.Board) -> chess.Move:
        """Ask Stockfish for a move in ``board``."""

        result = self.engine.play(board, self.limit)
        if result.move is None:
            raise RuntimeError("Stockfish returned no move")
        return result.move


def start_stockfish(binary_path: Path, config: StockfishConfig) -> chess.engine.SimpleEngine:
    """Start Stockfish and configure its Elo-limited UCI opponent."""

    engine = chess.engine.SimpleEngine.popen_uci(str(binary_path))
    try:
        engine.configure(
            {
                "UCI_LimitStrength": True,
                "UCI_Elo": config.elo,
                "Threads": config.threads,
                "Hash": config.hash_mb,
            }
        )
    except Exception:
        engine.quit()
        raise
    return engine


def ensure_stockfish_binary(binary_path: Path | None, cache_dir: Path) -> Path:
    """Return a usable binary, downloading the latest release when needed."""

    if binary_path is not None:
        _validate_binary(binary_path)
        return binary_path

    cached_binary = cache_dir / _cached_binary_name()
    if cached_binary.is_file() and os.access(cached_binary, os.X_OK):
        return cached_binary
    return download_stockfish(cache_dir)


def download_stockfish(cache_dir: Path) -> Path:
    """Download and extract the latest official Stockfish binary for this host."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    asset = _latest_platform_asset()
    with tempfile.TemporaryDirectory(prefix="stockfish-", dir=cache_dir) as temporary_name:
        temporary_dir = Path(temporary_name)
        archive_path = temporary_dir / asset.name
        _download_file(asset.download_url, archive_path)
        extracted_dir = temporary_dir / "extracted"
        extracted_dir.mkdir()
        _extract_archive(archive_path, extracted_dir)
        source_binary = _find_binary(extracted_dir)
        destination = cache_dir / _cached_binary_name()
        shutil.copy2(source_binary, destination)

    destination.chmod(destination.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return destination


def _latest_platform_asset() -> StockfishAsset:
    request = Request(
        STOCKFISH_RELEASES_API_URL,
        headers={"Accept": "application/vnd.github+json", "User-Agent": "pink-elephant"},
    )
    with urlopen(request, timeout=30) as response:
        loaded = json.loads(response.read())
    payload = _release_payload(loaded)
    assets = tuple(
        StockfishAsset(name=asset["name"], download_url=asset["browser_download_url"])
        for asset in payload["assets"]
    )
    return select_platform_asset(assets, system=platform.system(), machine=platform.machine())


def select_platform_asset(
    assets: Sequence[StockfishAsset], *, system: str, machine: str
) -> StockfishAsset:
    """Select the best release archive for a normalized host platform."""

    system_name = system.lower()
    machine_name = machine.lower()
    system_tokens = _system_tokens(system_name)
    machine_tokens = _machine_tokens(machine_name)
    candidates: list[tuple[int, StockfishAsset]] = []
    for asset in assets:
        name = asset.name.lower()
        if not _is_archive(name) or "src" in name or "source" in name:
            continue
        if not any(token in name for token in system_tokens):
            continue
        if machine_tokens and not any(token in name for token in machine_tokens):
            continue
        score = sum(token in name for token in system_tokens) * 10
        score += sum(token in name for token in machine_tokens)
        candidates.append((score, asset))

    if not candidates:
        names = ", ".join(sorted(asset.name for asset in assets))
        raise RuntimeError(
            f"no Stockfish release asset matches {system}/{machine}; available assets: {names}"
        )
    return max(candidates, key=lambda candidate: (candidate[0], candidate[1].name))[1]


def _release_payload(loaded: object) -> _ReleasePayload:
    if not isinstance(loaded, Mapping):
        raise RuntimeError("Stockfish release response was not a JSON object")
    raw_assets = loaded.get("assets")
    if not isinstance(raw_assets, list):
        raise RuntimeError("Stockfish release response did not contain an asset list")

    assets: list[_ReleaseAssetPayload] = []
    for raw_asset in raw_assets:
        if not isinstance(raw_asset, Mapping):
            raise RuntimeError("Stockfish release asset was not an object")
        name = raw_asset.get("name")
        download_url = raw_asset.get("browser_download_url")
        if not isinstance(name, str) or not isinstance(download_url, str):
            raise RuntimeError("Stockfish release asset had an invalid name or download URL")
        assets.append({"name": name, "browser_download_url": download_url})
    return cast(_ReleasePayload, {"assets": assets})


def _download_file(url: str, destination: Path) -> None:
    request = Request(url, headers={"User-Agent": "pink-elephant"})
    with urlopen(request, timeout=120) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)


def _extract_archive(archive_path: Path, destination: Path) -> None:
    if archive_path.name.lower().endswith(".zip"):
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                _validate_archive_member(member.filename, destination)
            archive.extractall(destination)
        return

    with tarfile.open(archive_path, mode="r:*") as archive:
        for member in archive.getmembers():
            _validate_archive_member(member.name, destination)
            if member.issym() or member.islnk():
                raise RuntimeError(f"refusing to extract archive link: {member.name}")
        archive.extractall(destination)


def _validate_archive_member(name: str, destination: Path) -> None:
    member_path = (destination / name).resolve()
    if destination.resolve() not in member_path.parents:
        raise RuntimeError(f"refusing to extract archive member outside cache: {name}")


def _find_binary(extracted_dir: Path) -> Path:
    candidates = tuple(
        path
        for path in sorted(extracted_dir.rglob("*"))
        if path.is_file()
        and path.name.lower().startswith("stockfish")
        and path.suffix.lower() not in {".exe", ".txt", ".md"}
    )
    if not candidates:
        raise RuntimeError("downloaded Stockfish archive did not contain an executable")
    return candidates[0]


def _validate_binary(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Stockfish binary does not exist: {path}")
    if not os.access(path, os.X_OK):
        raise PermissionError(f"Stockfish binary is not executable: {path}")


def _cached_binary_name() -> str:
    return "stockfish.exe" if platform.system().lower() == "windows" else "stockfish"


def _is_archive(name: str) -> bool:
    return name.endswith((".zip", ".tar", ".tar.gz", ".tgz"))


def _system_tokens(system: str) -> tuple[str, ...]:
    if system == "darwin":
        return ("macos", "mac", "osx")
    if system == "windows":
        return ("windows", "win")
    if system == "linux":
        return ("ubuntu", "linux")
    return (system,)


def _machine_tokens(machine: str) -> tuple[str, ...]:
    if machine in {"arm64", "aarch64"}:
        return ("arm64", "aarch64", "armv8", "m1", "apple-silicon")
    if machine in {"x86_64", "amd64"}:
        return ("x86-64", "x86_64", "amd64", "avx2", "sse41")
    return (machine,)
