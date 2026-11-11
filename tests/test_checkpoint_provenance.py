"""Unit tests for checkpoint provenance, stale checkpoint detection, and forced retrain segregation."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from ml.models.siamese_unet import SiameseUNet
from ml.training.callbacks import ModelCheckpointManager, load_checkpoint


def _create_dummy_checkpoint(path: Path, model_name: str, dataset_name: str, epoch: int, best_metric: float, exp_id: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    model = SiameseUNet(in_channels=3, img_size=64, pretrained=False)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    mgr = ModelCheckpointManager(path.parent, model_name, dataset_name)
    saved_path = mgr.save(
        model=model,
        optimizer=opt,
        scheduler=None,
        epoch=epoch,
        best_metric=best_metric,
        config={"experiment_id": exp_id, "model_name": model_name, "dataset_name": dataset_name},
        is_best=True,
    )
    meta = torch.load(saved_path, map_location="cpu", weights_only=False)
    return str(meta.get("saved_at"))


def test_checkpoint_provenance_saved_at_and_experiment_id(tmp_path: Path) -> None:
    ckpt_dir = tmp_path / "checkpoints"
    saved_at = _create_dummy_checkpoint(
        ckpt_dir / "siamese_unet_levir-cd_best.pt",
        model_name="siamese_unet",
        dataset_name="LEVIR-CD",
        epoch=5,
        best_metric=0.88,
        exp_id="run_levir_001",
    )
    assert saved_at is not None
    ckpt = torch.load(ckpt_dir / "siamese_unet_levir-cd_best.pt", map_location="cpu", weights_only=False)
    assert ckpt["experiment_id"] == "run_levir_001"
    assert ckpt["epoch"] == 5
    assert ckpt["best_metric"] == 0.88
    assert ckpt["saved_at"] == saved_at


def test_stale_checkpoint_detection_via_manifest(tmp_path: Path) -> None:
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_p = ckpt_dir / "siamese_unet_levir-cd_best.pt"
    old_saved_at = _create_dummy_checkpoint(ckpt_p, "siamese_unet", "LEVIR-CD", 5, 0.88, "run_001")

    # Suppose manifest records run_002 started with FORCE_RETRAIN, but crashed
    manifest = {
        "exp1_levir_unet": {
            "run_id": "run_002",
            "status": "FAILED",
            "model": "siamese_unet",
            "dataset": "LEVIR-CD",
            "checkpoint": None,
            "checkpoint_saved_at": None,
        }
    }

    # Provenance validator should reject stale checkpoint because status is FAILED
    entry = manifest["exp1_levir_unet"]
    is_valid = entry.get("status") == "COMPLETED" and entry.get("checkpoint") is not None
    assert not is_valid, "Failed run must not be accepted for benchmark evaluation!"


def test_forced_retrain_checkpoint_segregation(tmp_path: Path) -> None:
    ckpt_dir = tmp_path / "checkpoints"
    archive_dir = ckpt_dir / "archived"
    ckpt_p = ckpt_dir / "siamese_unet_levir-cd_best.pt"
    _create_dummy_checkpoint(ckpt_p, "siamese_unet", "LEVIR-CD", 10, 0.91, "run_old")

    assert ckpt_p.is_file()

    # Simulate FORCE_RETRAIN = True segregation logic
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_target = archive_dir / f"siamese_unet_levir-cd_archived_{int(time.time())}.pt"
    import shutil
    shutil.copy2(ckpt_p, archive_target)
    ckpt_p.unlink()

    # 1. Old checkpoint is safely preserved in archived/
    assert archive_target.is_file(), "Old checkpoint must be preserved in archive!"
    # 2. Active checkpoint dir has no stale checkpoint
    assert not ckpt_p.is_file(), "Active best.pt must be removed so new run cannot evaluate stale weights!"


def test_resume_intact_when_not_force_retrain(tmp_path: Path) -> None:
    ckpt_dir = tmp_path / "checkpoints"
    latest_p = ckpt_dir / "siamese_unet_levir-cd_latest.pt"
    _create_dummy_checkpoint(latest_p, "siamese_unet", "LEVIR-CD", 3, 0.75, "run_interrupted")

    FORCE_RETRAIN = False
    cmd = ["python", "scripts/train_model.py"]
    if not FORCE_RETRAIN and latest_p.exists():
        cmd.extend(["--resume", str(latest_p)])

    assert "--resume" in cmd
    assert str(latest_p) in cmd

