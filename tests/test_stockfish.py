from __future__ import annotations

import io
import json
import os
import tarfile
from pathlib import Path

import pytest

from pink_elephant.stockfish import (
    MAX_UCI_ELO,
    MIN_UCI_ELO,
    StockfishAsset,
    StockfishConfig,
    download_stockfish,
    ensure_stockfish_binary,
    select_platform_asset,
)


@pytest.mark.parametrize("elo", (MIN_UCI_ELO - 1, MAX_UCI_ELO + 1))
def test_stockfish_elo_must_be_supported(elo: int) -> None:
    with pytest.raises(ValueError, match="elo must be between"):
        StockfishConfig(elo=elo)


def test_stockfish_search_limit_prefers_movetime() -> None:
    assert StockfishConfig(movetime_ms=250).search_limit().time == 0.25


def test_platform_asset_selection_matches_host_architecture() -> None:
    assets = (
        StockfishAsset("stockfish-macos-m1-apple-silicon.tar", "arm-url"),
        StockfishAsset("stockfish-macos-x86-64.tar", "intel-url"),
        StockfishAsset("stockfish-ubuntu-x86-64-avx2.tar", "linux-url"),
    )

    selected = select_platform_asset(assets, system="Darwin", machine="arm64")

    assert selected.download_url == "arm-url"


def test_existing_executable_is_used_without_downloading(tmp_path: Path) -> None:
    binary = tmp_path / "stockfish"
    binary.write_bytes(b"binary")
    binary.chmod(0o700)

    assert ensure_stockfish_binary(binary, tmp_path / "cache") == binary


def test_download_stockfish_extracts_latest_platform_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as tar:
        content = b"fake stockfish"
        member = tarfile.TarInfo("stockfish-macos-m1-apple-silicon")
        member.mode = 0o755
        member.size = len(content)
        tar.addfile(member, io.BytesIO(content))
    archive_bytes = archive.getvalue()
    release_bytes = json.dumps(
        {
            "assets": [
                {
                    "name": "stockfish-macos-m1-apple-silicon.tar",
                    "browser_download_url": "https://example.test/stockfish.tar",
                }
            ]
        }
    ).encode()

    class Response:
        def __init__(self, content: bytes) -> None:
            self.content = content
            self.offset = 0

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self, size: int = -1) -> bytes:
            if size < 0:
                size = len(self.content) - self.offset
            result = self.content[self.offset : self.offset + size]
            self.offset += len(result)
            return result

    def fake_urlopen(request: object, *, timeout: int) -> Response:
        url = request.full_url  # type: ignore[attr-defined]
        return Response(release_bytes if "api.github.com" in url else archive_bytes)

    monkeypatch.setattr("pink_elephant.stockfish.urlopen", fake_urlopen)
    monkeypatch.setattr("pink_elephant.stockfish.platform.system", lambda: "Darwin")
    monkeypatch.setattr("pink_elephant.stockfish.platform.machine", lambda: "arm64")

    destination = download_stockfish(tmp_path / "cache")

    assert destination.read_bytes() == b"fake stockfish"
    assert os.access(destination, os.X_OK)
