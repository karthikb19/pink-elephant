"""Structured progress logging for self-play generation."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import TypeAlias

LogField: TypeAlias = bool | float | int | None | str


def configure_logging() -> None:
    """Configure INFO logs for local and Modal execution."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        force=False,
    )
    logging.getLogger("pink_elephant.self_play").setLevel(logging.INFO)


def log_event(logger: logging.Logger, event: str, fields: Mapping[str, LogField]) -> None:
    """Emit one JSON event that Modal captures as a searchable log line."""

    payload: dict[str, LogField] = {"event": event, **fields}
    logger.info(json.dumps(payload, sort_keys=True, separators=(",", ":")))
