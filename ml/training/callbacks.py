"""Checkpointing, early stopping, TensorBoard, and JSON/CSV experiment logging."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - optional fallback if tensorboard runtime absent
    SummaryWriter = None  # type: ignore[assignment]


class EarlyStopping:
    """Monitor a scalar validation metric and halt when improvement stalls."""

    def __init__(self, patience: int = 8, mode: str = "max", min_delta: float = 1e-4) -> None:
        self.patience = max(int(patience), 1)
        self.mode = mode
        self.min_delta = min_delta
        self.best_score: float | None = None
        self.counter = 0
        self.should_stop = False

    def step(self, value: float) -> bool:
        """Update state with current epoch metric. Returns True if new best achieved."""
        if self.best_score is None:
            self.best_score = value
            return True

        improved = (
            value > self.best_score + self.min_delta
            if self.mode == "max"
            else value < self.best_score - self.min_delta
        )
        if improved:
            self.best_score = value
            self.counter = 0
            return True

        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False


class ModelCheckpointManager:
    """Manages saving and resuming best & latest model checkpoints."""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        model_name: str,
        dataset_name: str,
        backup_dir: str | Path | None = None,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir = Path(backup_dir) if backup_dir else None
        if self.backup_dir:
            self.backup_dir.mkdir(parents=True, exist_ok=True)
        safe_ds = dataset_name.lower().replace(" ", "_")
        self.best_path = self.checkpoint_dir / f"{model_name}_{safe_ds}_best.pt"
        self.latest_path = self.checkpoint_dir / f"{model_name}_{safe_ds}_latest.pt"

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        epoch: int,
        best_metric: float,
        config: dict[str, Any],
        is_best: bool,
        training_status: str = "fine_tuned",
        scaler: Any = None,
    ) -> Path:
        import shutil

        payload = {
            "epoch": epoch,
            "best_metric": best_metric,
            "experiment_id": config.get("experiment_id"),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state_dict": scaler.state_dict() if scaler is not None and hasattr(scaler, "state_dict") else None,
            "model_info": model.model_info() if hasattr(model, "model_info") else {},
            "training_status": training_status,
            "config": config,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        torch.save(payload, self.latest_path)
        if self.backup_dir:
            try:
                shutil.copy2(self.latest_path, self.backup_dir / self.latest_path.name)
            except Exception as exc:
                print(f"Warning: checkpoint backup to {self.backup_dir} failed: {exc}")

        if is_best:
            torch.save(payload, self.best_path)
            if self.backup_dir:
                try:
                    shutil.copy2(self.best_path, self.backup_dir / self.best_path.name)
                except Exception as exc:
                    print(f"Warning: best checkpoint backup to {self.backup_dir} failed: {exc}")
            return self.best_path
        return self.latest_path


def load_checkpoint(
    checkpoint_path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    scaler: Any = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load model, optimizer, scheduler, and AMP scaler state from a saved checkpoint."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict, strict=True)

    if hasattr(model, "training_status"):
        model.training_status = ckpt.get("training_status", "fine_tuned")

    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler is not None and ckpt.get("scaler_state_dict") is not None and hasattr(scaler, "load_state_dict"):
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    return ckpt


class ExperimentLogger:
    """Persists training history to JSON, CSV, and TensorBoard logs."""

    def __init__(self, experiments_dir: str | Path, experiment_id: str) -> None:
        self.experiments_dir = Path(experiments_dir)
        self.experiment_id = experiment_id
        self.log_dir = self.experiments_dir / "logs" / experiment_id
        self.results_dir = self.experiments_dir / "results"
        self.configs_dir = self.experiments_dir / "configs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.configs_dir.mkdir(parents=True, exist_ok=True)

        self.history: list[dict[str, Any]] = []
        self.writer = SummaryWriter(log_dir=str(self.log_dir)) if SummaryWriter is not None else None

    def save_config(self, config: dict[str, Any]) -> Path:
        path = self.configs_dir / f"{self.experiment_id}_config.json"
        path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        return path

    def log_epoch(self, epoch: int, row: dict[str, Any]) -> None:
        entry = {"epoch": epoch, **row}
        self.history.append(entry)

        if self.writer is not None:
            for key, val in row.items():
                if isinstance(val, (int, float)):
                    self.writer.add_scalar(key, float(val), epoch)

        history_json = self.results_dir / f"{self.experiment_id}_history.json"
        history_json.write_text(json.dumps(self.history, indent=2), encoding="utf-8")

        history_csv = self.results_dir / f"{self.experiment_id}_history.csv"
        fieldnames = list(self.history[0].keys())
        with history_csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.history)

    def save_summary(self, summary: dict[str, Any]) -> Path:
        path = self.results_dir / f"{self.experiment_id}_summary.json"
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return path

    def close(self) -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()

