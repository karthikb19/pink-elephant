"""Render a dependency-free live dashboard for local training runs."""

# The generated HTML and inline JavaScript intentionally use long template lines.
# ruff: noqa: E501

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pink_elephant.contracts import ValidationMetrics
from pink_elephant.training import TrainingSummary


@dataclass(frozen=True)
class TrainingRunRecord:
    """Training and validation results for one completed epoch."""

    epoch: int
    step: int
    training: TrainingSummary | None
    validation: ValidationMetrics
    checkpoint: str | None = None
    elapsed_seconds: float | None = None

    def to_payload(self) -> dict[str, object]:
        """Return the JSON-compatible history representation."""

        return {
            "epoch": self.epoch,
            "step": self.step,
            "training": _training_payload(self.training),
            "validation": _validation_payload(self.validation),
            "checkpoint": self.checkpoint,
            "elapsed_seconds": self.elapsed_seconds,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> TrainingRunRecord:
        """Reconstruct one history record from JSON data."""

        validation_payload = payload.get("validation")
        if not isinstance(validation_payload, Mapping):
            raise ValueError("training history validation must be an object")
        training_payload = payload.get("training")
        training = None
        if training_payload is not None:
            if not isinstance(training_payload, Mapping):
                raise ValueError("training history training must be an object or null")
            training = _training_from_payload(training_payload)
        checkpoint = payload.get("checkpoint")
        if checkpoint is not None and not isinstance(checkpoint, str):
            raise ValueError("training history checkpoint must be a string or null")
        return cls(
            epoch=_required_int(payload, "epoch"),
            step=_required_int(payload, "step"),
            training=training,
            validation=_validation_from_payload(validation_payload),
            checkpoint=checkpoint,
            elapsed_seconds=_optional_float(payload, "elapsed_seconds"),
        )


def write_training_history(path: Path, records: Sequence[TrainingRunRecord]) -> None:
    """Write the complete JSON history used by the dashboard and other tools."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [record.to_payload() for record in records]
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_training_history(path: Path) -> tuple[TrainingRunRecord, ...]:
    """Read and validate a previously written training history."""

    decoded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(decoded, list):
        raise ValueError("training history must contain a JSON array")
    records: list[TrainingRunRecord] = []
    for item in decoded:
        if not isinstance(item, Mapping):
            raise ValueError("training history records must be JSON objects")
        records.append(TrainingRunRecord.from_payload(item))
    return tuple(records)


def write_training_dashboard(
    path: Path,
    records: Sequence[TrainingRunRecord],
    *,
    title: str = "RASnet training dashboard",
    target_epoch: int | None = None,
    refresh_seconds: int = 10,
) -> None:
    """Write a self-contained auto-refreshing SVG dashboard."""

    if refresh_seconds < 0:
        raise ValueError("refresh_seconds must be non-negative")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_training_dashboard(
            records,
            title=title,
            target_epoch=target_epoch,
            refresh_seconds=refresh_seconds,
        ),
        encoding="utf-8",
    )


def render_training_dashboard(
    records: Sequence[TrainingRunRecord],
    *,
    title: str = "RASnet training dashboard",
    target_epoch: int | None = None,
    refresh_seconds: int = 10,
) -> str:
    """Return a browser-ready dashboard with embedded training history."""

    if refresh_seconds < 0:
        raise ValueError("refresh_seconds must be non-negative")
    payload = json.dumps(
        [record.to_payload() for record in records],
        separators=(",", ":"),
    )
    target = "null" if target_epoch is None else str(target_epoch)
    generated_at = datetime.now(UTC).isoformat(timespec="seconds")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="dark">
  <title>{title}</title>
  <style>
    :root {{
      color-scheme: dark;
      --background: #111827;
      --panel: #1f2937;
      --panel-light: #273449;
      --text: #e5e7eb;
      --muted: #9ca3af;
      --grid: #374151;
      --blue: #60a5fa;
      --green: #34d399;
      --orange: #fbbf24;
      --pink: #f472b6;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--background);
      color: var(--text);
      font-family: ui-sans-serif, system-ui, -apple-system, sans-serif;
    }}
    main {{ max-width: 1400px; margin: 0 auto; padding: 28px; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    h2 {{ margin: 0 0 12px; font-size: 18px; }}
    p {{ color: var(--muted); margin: 6px 0; }}
    .header {{ display: flex; justify-content: space-between; gap: 20px; flex-wrap: wrap; }}
    .status {{ color: var(--green); font-weight: 700; }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 12px;
      margin: 24px 0;
    }}
    .card, .panel {{ background: var(--panel); border-radius: 12px; padding: 16px; }}
    .card-label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; }}
    .card-value {{ font-size: 24px; font-weight: 700; margin-top: 5px; }}
    .charts {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(440px, 1fr)); gap: 16px; }}
    .chart {{ width: 100%; height: auto; background: var(--panel-light); border-radius: 8px; }}
    .legend {{ display: flex; gap: 16px; flex-wrap: wrap; color: var(--muted); font-size: 12px; }}
    .legend span::before {{ content: ""; display: inline-block; width: 22px; border-top: 3px solid var(--color); margin: 0 6px 3px 0; }}
    .table-wrap {{ overflow-x: auto; margin-top: 16px; }}
    table {{ border-collapse: collapse; width: 100%; min-width: 960px; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid var(--grid); padding: 9px 8px; text-align: right; white-space: nowrap; }}
    th {{ color: var(--muted); font-weight: 600; }}
    th:first-child, td:first-child, th:last-child, td:last-child {{ text-align: left; }}
    .checkpoint {{ color: var(--green); }}
    code {{ color: var(--orange); }}
    @media (max-width: 700px) {{ main {{ padding: 16px; }} .charts {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
<main>
  <section class="header">
    <div>
      <h1>{title}</h1>
      <p>Live local dashboard · generated {generated_at} · refreshes every {refresh_seconds}s</p>
    </div>
    <p id="status" class="status"></p>
  </section>
  <section class="cards" id="cards"></section>
  <section class="charts">
    <article class="panel">
      <h2>Validation policy loss</h2>
      <svg id="policy-chart" class="chart" viewBox="0 0 900 320" role="img" aria-label="Validation policy loss chart"></svg>
      <div class="legend">
        <span style="--color: var(--blue)">Policy loss</span>
        <span style="--color: var(--orange)">Uniform-legal baseline</span>
      </div>
    </article>
    <article class="panel">
      <h2>Validation value error</h2>
      <svg id="value-chart" class="chart" viewBox="0 0 900 320" role="img" aria-label="Validation value error chart"></svg>
      <div class="legend">
        <span style="--color: var(--green)">MSE</span>
        <span style="--color: var(--pink)">MAE</span>
      </div>
    </article>
    <article class="panel">
      <h2>Policy accuracy</h2>
      <svg id="accuracy-chart" class="chart" viewBox="0 0 900 320" role="img" aria-label="Policy accuracy chart"></svg>
      <div class="legend">
        <span style="--color: var(--blue)">Top-1</span>
        <span style="--color: var(--green)">Top-5</span>
      </div>
    </article>
  </section>
  <section class="panel" style="margin-top: 16px">
    <h2>Epoch details</h2>
    <div class="table-wrap"><table>
      <thead><tr>
        <th>Epoch</th><th>Step</th><th>Train total</th><th>Train policy</th><th>Train value</th>
        <th>Val policy</th><th>Uniform</th><th>Top-1</th><th>Top-5</th><th>Val MSE</th><th>Val MAE</th><th>Checkpoint</th>
      </tr></thead>
      <tbody id="details"></tbody>
    </table></div>
  </section>
</main>
<script>
const history = {payload};
const targetEpoch = {target};
const refreshSeconds = {refresh_seconds};
const colors = ["#60a5fa", "#34d399", "#fbbf24", "#f472b6"];

function valueAt(record, path) {{
  return path.split(".").reduce((value, key) => value == null ? null : value[key], record);
}}

function formatNumber(value, digits = 4) {{
  return value == null ? "—" : Number(value).toFixed(digits);
}}

function drawChart(id, series, options = {{}}) {{
  const svg = document.getElementById(id);
  const width = 900;
  const height = 320;
  const pad = {{ left: 58, right: 22, top: 18, bottom: 38 }};
  const values = series.flatMap(item => history.map(record => valueAt(record, item.path)).filter(value => value != null));
  if (!values.length) {{ svg.innerHTML = "<text x='450' y='160' text-anchor='middle' fill='#9ca3af'>Waiting for metrics…</text>"; return; }}
  let min = options.min ?? Math.min(...values);
  let max = options.max ?? Math.max(...values);
  if (options.zero) min = 0;
  if (max === min) max = min + 1;
  const range = max - min;
  min = Math.max(options.min ?? -Infinity, min - range * 0.08);
  max = Math.min(options.max ?? Infinity, max + range * 0.08);
  const x = index => pad.left + (history.length === 1 ? 0.5 : index / (history.length - 1)) * (width - pad.left - pad.right);
  const y = value => height - pad.bottom - ((value - min) / (max - min)) * (height - pad.top - pad.bottom);
  let markup = "";
  for (let tick = 0; tick <= 4; tick++) {{
    const tickValue = min + (max - min) * tick / 4;
    const tickY = y(tickValue);
    markup += `<line x1='${{pad.left}}' x2='${{width - pad.right}}' y1='${{tickY}}' y2='${{tickY}}' stroke='#374151'/>`;
    markup += `<text x='${{pad.left - 8}}' y='${{tickY + 4}}' text-anchor='end' fill='#9ca3af' font-size='12'>${{tickValue.toFixed(2)}}</text>`;
  }}
  markup += `<line x1='${{pad.left}}' x2='${{width - pad.right}}' y1='${{height - pad.bottom}}' y2='${{height - pad.bottom}}' stroke='#6b7280'/>`;
  history.forEach((record, index) => {{
    const label = record.epoch;
    markup += `<text x='${{x(index)}}' y='${{height - 12}}' text-anchor='middle' fill='#9ca3af' font-size='12'>${{label}}</text>`;
  }});
  series.forEach((item, seriesIndex) => {{
    const points = history.map((record, index) => {{
      const value = valueAt(record, item.path);
      return value == null ? null : `${{x(index)}},${{y(value)}}`;
    }}).filter(point => point != null).join(" ");
    markup += `<polyline points='${{points}}' fill='none' stroke='${{colors[seriesIndex]}}' stroke-width='3' stroke-linejoin='round' stroke-linecap='round'/>`;
  }});
  svg.innerHTML = markup;
}}

function render() {{
  const latest = history[history.length - 1];
  const validation = latest?.validation;
  const complete = targetEpoch != null && latest?.epoch >= targetEpoch;
  document.getElementById("status").textContent = complete ? `Complete · epoch ${{latest.epoch}}` : `Running · epoch ${{latest?.epoch ?? 0}}${{targetEpoch ? ` / ${{targetEpoch}}` : ""}}`;
  const cards = [
    ["Epoch", latest?.epoch ?? 0],
    ["Steps", latest?.step ?? 0],
    ["Val policy loss", formatNumber(validation?.policy_loss)],
    ["Uniform baseline", formatNumber(validation?.uniform_policy_loss)],
    ["Top-1 accuracy", validation == null ? "—" : `${{(validation.policy_top1_accuracy * 100).toFixed(2)}}%`],
    ["Top-5 accuracy", validation == null ? "—" : `${{(validation.policy_top5_accuracy * 100).toFixed(2)}}%`],
  ];
  document.getElementById("cards").innerHTML = cards.map(card => `<div class='card'><div class='card-label'>${{card[0]}}</div><div class='card-value'>${{card[1]}}</div></div>`).join("");
  drawChart("policy-chart", [
    {{ path: "validation.policy_loss" }},
    {{ path: "validation.uniform_policy_loss" }},
  ], {{ zero: true }});
  drawChart("value-chart", [
    {{ path: "validation.value_mse" }},
    {{ path: "validation.value_mae" }},
  ], {{ zero: true }});
  drawChart("accuracy-chart", [
    {{ path: "validation.policy_top1_accuracy" }},
    {{ path: "validation.policy_top5_accuracy" }},
  ], {{ min: 0, max: 1 }});
  document.getElementById("details").innerHTML = history.map(record => {{
    const train = record.training;
    const val = record.validation;
    const checkpoint = record.checkpoint ? `<span class='checkpoint'>${{record.checkpoint}}</span>` : "—";
    return `<tr><td>${{record.epoch}}</td><td>${{record.step}}</td><td>${{formatNumber(train?.total_loss)}}</td><td>${{formatNumber(train?.policy_loss)}}</td><td>${{formatNumber(train?.value_loss)}}</td><td>${{formatNumber(val.policy_loss)}}</td><td>${{formatNumber(val.uniform_policy_loss)}}</td><td>${{(val.policy_top1_accuracy * 100).toFixed(2)}}%</td><td>${{(val.policy_top5_accuracy * 100).toFixed(2)}}%</td><td>${{formatNumber(val.value_mse)}}</td><td>${{formatNumber(val.value_mae)}}</td><td>${{checkpoint}}</td></tr>`;
  }}).join("");
}}
render();
if (refreshSeconds > 0 && !(targetEpoch != null && history[history.length - 1]?.epoch >= targetEpoch)) {{
  setTimeout(() => location.reload(), refreshSeconds * 1000);
}}
</script>
</body>
</html>
"""


def _training_payload(training: TrainingSummary | None) -> dict[str, float | int] | None:
    if training is None:
        return None
    return {
        "example_count": training.example_count,
        "total_loss": training.total_loss,
        "policy_loss": training.policy_loss,
        "value_loss": training.value_loss,
    }


def _validation_payload(metrics: ValidationMetrics) -> dict[str, float | int]:
    return {
        "example_count": metrics.example_count,
        "policy_loss": metrics.policy_loss,
        "uniform_policy_loss": metrics.uniform_policy_loss,
        "policy_top1_accuracy": metrics.policy_top1_accuracy,
        "policy_top5_accuracy": metrics.policy_top5_accuracy,
        "value_mse": metrics.value_mse,
        "value_mae": metrics.value_mae,
    }


def _training_from_payload(payload: Mapping[str, object]) -> TrainingSummary:
    return TrainingSummary(
        example_count=_required_int(payload, "example_count"),
        total_loss=_required_float(payload, "total_loss"),
        policy_loss=_required_float(payload, "policy_loss"),
        value_loss=_required_float(payload, "value_loss"),
    )


def _validation_from_payload(payload: Mapping[str, object]) -> ValidationMetrics:
    return ValidationMetrics(
        example_count=_required_int(payload, "example_count"),
        policy_loss=_required_float(payload, "policy_loss"),
        uniform_policy_loss=_required_float(payload, "uniform_policy_loss"),
        policy_top1_accuracy=_required_float(payload, "policy_top1_accuracy"),
        policy_top5_accuracy=_required_float(payload, "policy_top5_accuracy"),
        value_mse=_required_float(payload, "value_mse"),
        value_mae=_required_float(payload, "value_mae"),
    )


def _required_int(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"training history field {key!r} must be an integer")
    return value


def _required_float(payload: Mapping[str, object], key: str) -> float:
    value = payload.get(key)
    if not isinstance(value, (float, int)) or isinstance(value, bool):
        raise ValueError(f"training history field {key!r} must be numeric")
    return float(value)


def _optional_float(payload: Mapping[str, object], key: str) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, (float, int)) or isinstance(value, bool):
        raise ValueError(f"training history field {key!r} must be numeric or null")
    return float(value)
