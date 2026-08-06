"""Stable run directories, manifests, and checkpoint naming."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, TypeAlias, TypedDict

from pink_elephant.model_adapter import ModelSpec, ModelSpecPayload

RUN_FORMAT_VERSION: Final[str] = "pink-elephant-run/v1"
DEFAULT_RUNS_ROOT: Final[Path] = Path("data/runs")
RunParameterValue: TypeAlias = str | int | float | bool | None
_RUN_ID_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^(?P<timestamp>\d{8}T\d{6}Z)-(?P<name>[a-z0-9]+(?:-[a-z0-9]+)*)$"
)
_CHECKPOINT_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^(?P<run_id>.+)-epoch-(?P<epoch>\d{6})-step-(?P<step>\d{9})\.pt$"
)


class RunManifestPayload(TypedDict):
    """JSON representation of one immutable run identity."""

    format_version: str
    run_id: str
    run_name: str
    created_at: str
    model: ModelSpecPayload
    parameters: dict[str, RunParameterValue]


@dataclass(frozen=True, slots=True)
class RunParameter:
    """One named scalar recorded for reproducible execution."""

    name: str
    value: RunParameterValue

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("run parameter name must not be empty")
        if isinstance(self.value, float) and not math.isfinite(self.value):
            raise ValueError(f"run parameter {self.name!r} must be finite")


@dataclass(frozen=True, slots=True)
class RunIdentity:
    """A human label paired with a sortable UTC run identifier."""

    run_id: str
    run_name: str
    created_at: datetime

    @classmethod
    def create(cls, run_name: str, *, created_at: datetime | None = None) -> RunIdentity:
        """Create a timestamp-prefixed run identity from a human label."""

        timestamp = created_at or datetime.now(UTC)
        if timestamp.tzinfo is None:
            raise ValueError("run creation time must include a timezone")
        utc_timestamp = timestamp.astimezone(UTC).replace(microsecond=0)
        normalized_name = normalize_run_name(run_name)
        run_id = f"{utc_timestamp.strftime('%Y%m%dT%H%M%SZ')}-{normalized_name}"
        return cls(run_id=run_id, run_name=normalized_name, created_at=utc_timestamp)

    @classmethod
    def parse(cls, run_id: str) -> RunIdentity:
        """Reconstruct an identity from its canonical identifier."""

        match = _RUN_ID_PATTERN.fullmatch(run_id)
        if match is None:
            raise ValueError("run_id must look like 20260806T012345Z-experiment-name")
        created_at = datetime.strptime(match.group("timestamp"), "%Y%m%dT%H%M%SZ").replace(
            tzinfo=UTC
        )
        return cls(run_id=run_id, run_name=match.group("name"), created_at=created_at)


@dataclass(frozen=True, slots=True)
class RunManifest:
    """Portable metadata stored at the root of every run directory."""

    identity: RunIdentity
    model: ModelSpec
    parameters: tuple[RunParameter, ...] = ()

    def __post_init__(self) -> None:
        names = tuple(parameter.name for parameter in self.parameters)
        if len(names) != len(set(names)):
            raise ValueError("run parameter names must be unique")

    def to_payload(self) -> RunManifestPayload:
        """Return the JSON representation of this manifest."""

        return {
            "format_version": RUN_FORMAT_VERSION,
            "run_id": self.identity.run_id,
            "run_name": self.identity.run_name,
            "created_at": self.identity.created_at.isoformat(),
            "model": self.model.to_payload(),
            "parameters": {parameter.name: parameter.value for parameter in self.parameters},
        }

    @classmethod
    def from_payload(cls, raw_payload: object) -> RunManifest:
        """Validate and reconstruct a run manifest."""

        if not isinstance(raw_payload, dict):
            raise ValueError("run manifest must be a JSON object")
        if raw_payload.get("format_version") != RUN_FORMAT_VERSION:
            raise ValueError("unsupported run manifest format")
        run_id = raw_payload.get("run_id")
        run_name = raw_payload.get("run_name")
        created_at = raw_payload.get("created_at")
        if not isinstance(run_id, str):
            raise ValueError("run manifest run_id must be a string")
        if not isinstance(run_name, str):
            raise ValueError("run manifest run_name must be a string")
        if not isinstance(created_at, str):
            raise ValueError("run manifest created_at must be a string")
        identity = RunIdentity.parse(run_id)
        parsed_created_at = datetime.fromisoformat(created_at)
        if parsed_created_at.tzinfo is None:
            raise ValueError("run manifest created_at must include a timezone")
        if run_name != identity.run_name:
            raise ValueError("run manifest name does not match run_id")
        if parsed_created_at.astimezone(UTC) != identity.created_at:
            raise ValueError("run manifest timestamp does not match run_id")
        raw_parameters = raw_payload.get("parameters")
        if not isinstance(raw_parameters, dict):
            raise ValueError("run manifest parameters must be a JSON object")
        parameters: list[RunParameter] = []
        for name, value in raw_parameters.items():
            if not isinstance(name, str):
                raise ValueError("run manifest parameter names must be strings")
            if value is not None and not isinstance(value, (str, int, float, bool)):
                raise ValueError(f"run manifest parameter {name!r} must be scalar")
            parameters.append(RunParameter(name, value))
        return cls(
            identity=identity,
            model=ModelSpec.from_payload(raw_payload.get("model")),
            parameters=tuple(sorted(parameters, key=lambda parameter: parameter.name)),
        )


@dataclass(frozen=True, slots=True)
class CheckpointStore:
    """Immutable checkpoint files belonging to exactly one run."""

    directory: Path
    run_id: str

    def __post_init__(self) -> None:
        RunIdentity.parse(self.run_id)

    def create(self) -> None:
        """Create the checkpoint directory, refusing an existing store."""

        self.directory.mkdir(parents=False, exist_ok=False)

    def path_for(self, epoch: int, step: int) -> Path:
        """Return the canonical path for one training position."""

        if epoch < 0:
            raise ValueError("checkpoint epoch must be non-negative")
        if step < 0:
            raise ValueError("checkpoint step must be non-negative")
        return self.directory / (f"{self.run_id}-epoch-{epoch:06d}-step-{step:09d}.pt")

    def list(self) -> tuple[Path, ...]:
        """Return checkpoints ordered by epoch and optimizer step."""

        if not self.directory.is_dir():
            return ()
        checkpoints: list[tuple[int, int, Path]] = []
        for path in self.directory.glob("*.pt"):
            match = _CHECKPOINT_PATTERN.fullmatch(path.name)
            if match is None or match.group("run_id") != self.run_id:
                continue
            checkpoints.append((int(match.group("epoch")), int(match.group("step")), path))
        return tuple(path for _, _, path in sorted(checkpoints))

    def resolve(self, selector: str = "latest") -> Path:
        """Resolve ``latest`` or one exact canonical checkpoint filename."""

        checkpoints = self.list()
        if selector == "latest":
            if not checkpoints:
                raise FileNotFoundError(f"run has no checkpoints: {self.run_id}")
            return checkpoints[-1]
        if Path(selector).name != selector:
            raise ValueError("checkpoint selector must be a filename or 'latest'")
        selected = self.directory / selector
        if selected not in checkpoints:
            raise FileNotFoundError(f"checkpoint {selector!r} does not exist in run {self.run_id}")
        return selected


@dataclass(frozen=True, slots=True)
class RunLayout:
    """Paths and manifest for one standardized local or mounted run."""

    root: Path
    manifest: RunManifest

    @property
    def directory(self) -> Path:
        """Return the run's root directory."""

        return self.root / self.manifest.identity.run_id

    @property
    def manifest_path(self) -> Path:
        """Return the run-manifest path."""

        return self.directory / "run.json"

    @property
    def checkpoints(self) -> CheckpointStore:
        """Return this run's checkpoint store."""

        return CheckpointStore(self.directory / "checkpoints", self.manifest.identity.run_id)

    @property
    def metrics_path(self) -> Path:
        """Return the latest-metrics path."""

        return self.directory / "metrics.json"

    @property
    def metrics_history_path(self) -> Path:
        """Return the append-only metrics-history path."""

        return self.directory / "metrics-history.jsonl"

    @property
    def evaluations_directory(self) -> Path:
        """Return the directory for persisted arena results."""

        return self.directory / "evaluations"


class RunStore:
    """Create, discover, and open standardized run directories."""

    def __init__(self, root: Path = DEFAULT_RUNS_ROOT) -> None:
        self.root = root

    def create(
        self,
        run_name: str,
        model: ModelSpec,
        *,
        created_at: datetime | None = None,
        parameters: tuple[RunParameter, ...] = (),
    ) -> RunLayout:
        """Create an immutable run directory and manifest."""

        identity = RunIdentity.create(run_name, created_at=created_at)
        return self.initialize(identity, model, parameters=parameters)

    def initialize(
        self,
        identity: RunIdentity,
        model: ModelSpec,
        *,
        parameters: tuple[RunParameter, ...] = (),
    ) -> RunLayout:
        """Create a run from an identity already resolved by an execution client."""

        layout = RunLayout(
            self.root,
            RunManifest(
                identity=identity,
                model=model,
                parameters=tuple(sorted(parameters, key=lambda parameter: parameter.name)),
            ),
        )
        self.root.mkdir(parents=True, exist_ok=True)
        layout.directory.mkdir(exist_ok=False)
        layout.checkpoints.create()
        with layout.manifest_path.open("x", encoding="utf-8") as destination:
            json.dump(layout.manifest.to_payload(), destination, indent=2, sort_keys=True)
            destination.write("\n")
        return layout

    def open(self, run_id: str) -> RunLayout:
        """Load and validate one run by its exact identifier."""

        RunIdentity.parse(run_id)
        manifest_path = self.root / run_id / "run.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"run manifest does not exist: {manifest_path}")
        manifest = RunManifest.from_payload(json.loads(manifest_path.read_text(encoding="utf-8")))
        if manifest.identity.run_id != run_id:
            raise ValueError("run manifest identity does not match its directory")
        return RunLayout(self.root, manifest)

    def list(self) -> tuple[RunLayout, ...]:
        """Return valid runs ordered by creation time and name."""

        if not self.root.is_dir():
            return ()
        layouts: list[RunLayout] = []
        for manifest_path in self.root.glob("*/run.json"):
            layouts.append(self.open(manifest_path.parent.name))
        return tuple(sorted(layouts, key=lambda layout: layout.manifest.identity.run_id))


def normalize_run_name(run_name: str) -> str:
    """Convert a human run name into a safe, readable slug."""

    normalized = re.sub(r"[^a-z0-9]+", "-", run_name.strip().lower()).strip("-")
    if not normalized:
        raise ValueError("run name must contain at least one letter or number")
    return normalized
