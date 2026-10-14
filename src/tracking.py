"""MLflow tracking shared by the training scripts.

Everything is stored locally inside the project:
    mlflow.db     - run metadata and the model registry (SQLite)
    mlartifacts/  - logged files (results, artifacts, models)

Browse the runs from the project root with:
    mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
"""
from __future__ import annotations

from typing import Callable

import mlflow

from src import config

DB_PATH = config.PROJECT_ROOT / "mlflow.db"
ARTIFACT_ROOT = config.PROJECT_ROOT / "mlartifacts"
TRACKING_URI = f"sqlite:///{DB_PATH.as_posix()}"


def setup(experiment_name: str) -> None:
    """Point MLflow at the local store and select (or create) the experiment."""
    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(
            experiment_name,
            artifact_location=(ARTIFACT_ROOT / experiment_name).as_uri(),
        )
    mlflow.set_experiment(experiment_name)


def as_param(value):
    """MLflow params are strings: join lists, keep scalars as they are."""
    if isinstance(value, (list, tuple)):
        return ",".join(str(v) for v in value)
    return value


def log_bearing_metrics(results: dict) -> None:
    """Log '<bearing>/<metric>' for every test bearing, plus 'mean/<metric>'."""
    per_metric: dict[str, list[float]] = {}
    for bearing, metrics in results.items():
        for key, value in metrics.items():
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            mlflow.log_metric(f"{bearing}/{key}", value)
            per_metric.setdefault(key, []).append(value)
    for key, values in per_metric.items():
        mlflow.log_metric(f"mean/{key}", sum(values) / len(values))


def epoch_logger() -> Callable[[int, float, float], None]:
    """Callback for train_model: logs train and validation loss at every epoch."""
    def _log(epoch: int, train_loss: float, val_loss: float) -> None:
        mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)
    return _log
