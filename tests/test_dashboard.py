from pathlib import Path

from pink_elephant.contracts import ValidationMetrics
from pink_elephant.dashboard import (
    TrainingRunRecord,
    read_training_history,
    render_training_dashboard,
    write_training_dashboard,
    write_training_history,
)
from pink_elephant.training import TrainingSummary


def _record(epoch: int = 1) -> TrainingRunRecord:
    return TrainingRunRecord(
        epoch=epoch,
        step=epoch * 10,
        training=TrainingSummary(epoch, 2.0, 1.5, 0.5),
        validation=ValidationMetrics(epoch, 1.2, 2.0, 0.2, 0.7, 0.8, 0.6),
        checkpoint=f"epoch-{epoch:06d}.pt" if epoch % 2 == 0 else None,
        elapsed_seconds=3.5,
    )


def test_training_history_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "metrics.json"
    records = (_record(), _record(2))

    write_training_history(path, records)

    assert read_training_history(path) == records


def test_dashboard_contains_live_charts_and_epoch_details() -> None:
    html = render_training_dashboard((_record(), _record(2)), target_epoch=10)

    assert "policy-chart" in html
    assert "value-chart" in html
    assert "accuracy-chart" in html
    assert "setTimeout(() => location.reload()" in html
    assert "epoch-000002.pt" in html
    assert "targetEpoch = 10" in html


def test_dashboard_writer_creates_parent_directory(tmp_path: Path) -> None:
    path = tmp_path / "run" / "index.html"

    write_training_dashboard(path, (_record(),), refresh_seconds=0)

    assert path.exists()
    assert "refreshes every 0s" in path.read_text(encoding="utf-8")
