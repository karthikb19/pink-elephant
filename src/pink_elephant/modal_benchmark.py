"""Isolate model forward throughput on one L4, independent of self-play.

Six self-play runs have inferred GPU behaviour through a pipeline that also
contains a Rust search, Python revalidation, Arrow writes, and a drain tail. That
is enough moving parts that a slower container looks like a bad optimization: the
autocast round regressed the Rust engine by 71% per leaf, which autocast cannot
possibly cause.

This measures the model alone. It answers two questions the sweeps could only
guess at: where forward throughput saturates with batch size, and whether FP16
autocast or `torch.compile` actually help at the shapes self-play uses.

    uv run modal run src/pink_elephant/modal_benchmark.py
    uv run modal run src/pink_elephant/modal_benchmark.py --batch-sizes 64,256 --modes fp32,autocast
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import nullcontext
from typing import Final

import modal
import torch
from torch import nn

from pink_elephant.encoding import BOARD_SIZE, HALFMOVE_PLANE, HALFMOVE_SCALE, PLANE_COUNT
from pink_elephant.modal_image import build_image
from pink_elephant.model import ChessResNet, ModelOutput, ResNetConfig

logger = logging.getLogger(__name__)

BENCHMARK_GPU: Final[str] = "L4"
BENCHMARK_CPU: Final[float] = 2.0
BENCHMARK_TIMEOUT_SECONDS: Final[int] = 30 * 60
DEFAULT_BATCH_SIZES: Final[str] = "16,32,64,128,256,512"
DEFAULT_MODES: Final[str] = "fp32,autocast,compile,autocast+compile"

app = modal.App(name="pink-elephant-model-benchmark", image=build_image())


def _autocast_context(enabled: bool, device: torch.device):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _forward(model: nn.Module, inputs: torch.Tensor) -> ModelOutput:
    output = model(inputs)
    if not isinstance(output, ModelOutput):
        raise TypeError("benchmark model must return ModelOutput")
    return output


def _prepare(staging: torch.Tensor, count: int, device: torch.device) -> torch.Tensor:
    """Reproduce the host loop's input path exactly, including the uint8 transfer."""

    inputs = staging[:count].to(device, non_blocking=True).float()
    inputs[:, HALFMOVE_PLANE] /= HALFMOVE_SCALE
    return inputs


def measure(
    *,
    channels: int,
    residual_blocks: int,
    batch_sizes: tuple[int, ...],
    modes: tuple[str, ...],
    iterations: int,
    warmup: int,
    device_name: str,
) -> list[dict[str, object]]:
    """Time forward passes for every mode and batch size on one device."""

    device = torch.device(device_name)
    results: list[dict[str, object]] = []

    for mode in modes:
        autocast = "autocast" in mode
        compiled = "compile" in mode
        if device.type != "cuda" and (autocast or compiled):
            continue

        for batch in batch_sizes:
            torch.manual_seed(0)
            model = (
                ChessResNet(ResNetConfig(channels=channels, residual_blocks=residual_blocks))
                .to(device)
                .eval()
            )
            if compiled:
                model = torch.compile(model, dynamic=None)

            staging = torch.randint(
                0,
                2,
                (batch, PLANE_COUNT, BOARD_SIZE, BOARD_SIZE),
                dtype=torch.uint8,
                pin_memory=device.type == "cuda",
            )

            # Warm up cuDNN autotuning and any compilation before timing.
            with torch.inference_mode(), _autocast_context(autocast, device):
                for _ in range(warmup):
                    output = _forward(model, _prepare(staging, batch, device))
                    output.policy_logits.detach().to("cpu", dtype=torch.float32)
            if device.type == "cuda":
                torch.cuda.synchronize(device)

            # Forward only, measured with CUDA events so host overhead is excluded.
            if device.type == "cuda":
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                inputs = _prepare(staging, batch, device)
                # Warm up on this exact tensor. `torch.compile` specializes on the
                # input it traced, so timing a tensor the warmup never saw captures
                # a recompilation inside the event window and reports a forward
                # slower than the round trip that contains it.
                with torch.inference_mode(), _autocast_context(autocast, device):
                    for _ in range(warmup):
                        _forward(model, inputs)
                torch.cuda.synchronize(device)
                start.record()
                with torch.inference_mode(), _autocast_context(autocast, device):
                    for _ in range(iterations):
                        _forward(model, inputs)
                end.record()
                torch.cuda.synchronize(device)
                forward_ms = start.elapsed_time(end) / iterations
            else:
                started = time.perf_counter()
                inputs = _prepare(staging, batch, device)
                with torch.inference_mode():
                    for _ in range(iterations):
                        _forward(model, inputs)
                forward_ms = (time.perf_counter() - started) / iterations * 1_000

            # Full host round trip: the transfer, normalization, forward, and the
            # 4,672-wide logit copy back that the engine actually consumes.
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            with torch.inference_mode(), _autocast_context(autocast, device):
                for _ in range(iterations):
                    output = _forward(model, _prepare(staging, batch, device))
                    output.policy_logits.detach().to("cpu", dtype=torch.float32).numpy()
                    output.value.detach().to("cpu", dtype=torch.float32).numpy()
            roundtrip_ms = (time.perf_counter() - started) / iterations * 1_000

            results.append(
                {
                    "mode": mode,
                    "batch": batch,
                    "forward_ms": forward_ms,
                    "roundtrip_ms": roundtrip_ms,
                    "forward_positions_per_second": batch / forward_ms * 1_000,
                    "roundtrip_positions_per_second": batch / roundtrip_ms * 1_000,
                    "forward_microseconds_per_position": forward_ms * 1_000 / batch,
                }
            )
            del model, staging
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return results


@app.function(
    gpu=BENCHMARK_GPU,
    cpu=BENCHMARK_CPU,
    timeout=BENCHMARK_TIMEOUT_SECONDS,
    retries=0,
)
def benchmark_modal(
    channels: int,
    residual_blocks: int,
    batch_sizes: tuple[int, ...],
    modes: tuple[str, ...],
    iterations: int,
    warmup: int,
) -> list[dict[str, object]]:
    """Run the sweep on one L4 and return raw measurements."""

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info(
        json.dumps(
            {
                "event": "benchmark_started",
                "gpu": torch.cuda.get_device_name(0),
                "torch": torch.__version__,
                "channels": channels,
                "residual_blocks": residual_blocks,
            }
        )
    )
    return measure(
        channels=channels,
        residual_blocks=residual_blocks,
        batch_sizes=batch_sizes,
        modes=modes,
        iterations=iterations,
        warmup=warmup,
        device_name="cuda",
    )


def render(results: list[dict[str, object]], simulations: int) -> str:
    """Render one table per mode, with the self-play rate each throughput implies."""

    lines = [
        "",
        f"{'mode':16s}{'batch':>7s}{'fwd ms':>9s}{'fwd pos/s':>12s}"
        f"{'trip ms':>9s}{'trip pos/s':>12s}{'us/pos':>9s}{'moves/s':>10s}",
        "-" * 84,
    ]
    baseline: dict[int, float] = {}
    for row in results:
        mode, batch = str(row["mode"]), int(row["batch"])
        trip = float(row["roundtrip_positions_per_second"])
        if mode == "fp32":
            baseline[batch] = trip
        # Leaves per second divided by the simulation budget is the ceiling this
        # forward throughput places on recorded positions per second.
        lines.append(
            f"{mode:16s}{batch:7d}{float(row['forward_ms']):9.2f}"
            f"{float(row['forward_positions_per_second']):12,.0f}"
            f"{float(row['roundtrip_ms']):9.2f}{trip:12,.0f}"
            f"{float(row['forward_microseconds_per_position']):9.2f}"
            f"{trip / simulations:10,.0f}"
        )
    if baseline:
        lines += ["", "speedup versus fp32 at the same batch (round trip):"]
        for row in results:
            mode, batch = str(row["mode"]), int(row["batch"])
            if mode == "fp32" or batch not in baseline:
                continue
            ratio = float(row["roundtrip_positions_per_second"]) / baseline[batch]
            lines.append(f"  {mode:16s} batch {batch:4d}   {ratio:5.2f}x")
    return "\n".join(lines)


@app.local_entrypoint()
def main(
    channels: int = 192,
    residual_blocks: int = 12,
    batch_sizes: str = DEFAULT_BATCH_SIZES,
    modes: str = DEFAULT_MODES,
    iterations: int = 200,
    warmup: int = 30,
    simulations: int = 32,
) -> None:
    """Sweep batch sizes and precision modes on one L4."""

    sizes = tuple(int(value) for value in batch_sizes.split(",") if value.strip())
    selected = tuple(value.strip() for value in modes.split(",") if value.strip())
    results = benchmark_modal.remote(channels, residual_blocks, sizes, selected, iterations, warmup)
    print(render(results, simulations))
    print()
    print(json.dumps(results, indent=2, sort_keys=True))
