"""Quantitative evaluation and fair 3-architecture benchmarking engine."""

from __future__ import annotations

import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ml.evaluation.metrics import MetricAccumulator
from ml.models import MODEL_REGISTRY, get_model
from ml.training.callbacks import load_checkpoint
from ml.utils.device import detect_device, get_peak_memory_mb, reset_peak_memory_stats


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: str | torch.device = "auto",
    threshold: float = 0.5,
    checkpoint_path: str | Path | None = None,
    dataset_name: str = "unknown",
    split: str = "test",
    training_duration_sec: float | None = None,
    is_synthetic: bool = False,
) -> dict[str, Any]:
    """Evaluate a single change-detection model on a standardized DataLoader."""
    dev = detect_device(str(device)) if isinstance(device, str) else device
    model = model.to(dev)
    model.eval()

    ckpt_size_mb: float | None = None
    if checkpoint_path is not None:
        ckpt_p = Path(checkpoint_path)
        ckpt_meta = load_checkpoint(ckpt_p, model, map_location=dev)
        ckpt_size_mb = round(ckpt_p.stat().st_size / (1024.0 * 1024.0), 3)
        training_status = str(ckpt_meta.get("training_status", "fine_tuned"))
    else:
        training_status = str(getattr(model, "training_status", "untrained"))

    reset_peak_memory_stats(dev)
    acc = MetricAccumulator(threshold=threshold)
    num_samples = 0
    t0 = time.perf_counter()

    with torch.no_grad():
        for batch in data_loader:
            before = batch["before"].to(dev)
            after = batch["after"].to(dev)
            mask = batch["mask"].to(dev)
            valid_mask = batch.get("valid_mask")
            if valid_mask is not None:
                valid_mask = valid_mask.to(dev)

            logits = model(before, after)
            acc.update(logits, mask, valid_mask=valid_mask, from_logits=True)
            num_samples += int(before.shape[0])

    total_time = time.perf_counter() - t0
    metrics = acc.compute()
    info = model.model_info() if hasattr(model, "model_info") else {}
    ms_per_sample = (total_time * 1000.0 / num_samples) if num_samples > 0 else 0.0

    return {
        "model_name": info.get("model_name", type(model).__name__),
        "display_name": MODEL_REGISTRY.get(info.get("model_name", ""), {}).get(
            "display_name", info.get("model_name", type(model).__name__)
        ),
        "backbone_name": info.get("backbone_name", "unknown"),
        "backbone_weights_source": info.get("backbone_weights_source", "unknown"),
        "training_status": training_status,
        "dataset_name": dataset_name,
        "split": split,
        "is_synthetic": is_synthetic,
        "threshold": threshold,
        "num_samples": num_samples,
        "device": str(dev),
        "metrics": metrics.to_dict(),
        "confusion_matrix": metrics.confusion_matrix(),
        "total_parameters": info.get("total_parameters"),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "checkpoint_size_mb": ckpt_size_mb,
        "training_duration_sec": training_duration_sec,
        "inference_time_sec": round(total_time, 4),
        "inference_ms_per_sample": round(ms_per_sample, 3),
        "peak_memory_mb": get_peak_memory_mb(dev),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }


def benchmark_models(
    model_configs: Sequence[dict[str, Any]],
    data_loader: DataLoader,
    dataset_name: str,
    split: str = "test",
    device: str = "auto",
    threshold: float = 0.5,
    experiments_dir: str | Path = "experiments",
    is_synthetic: bool = False,
) -> dict[str, Any]:
    """Benchmark multiple architectures under identical data, splits, and thresholds.

    Each entry in ``model_configs`` accepts:
        ``{"model_name": "siamese_unet"|"vit"|"swin", "checkpoint_path": ...,
           "backbone_name": ..., "in_channels": 3, "img_size": 256,
           "pretrained": False, "training_duration_sec": ...}``
    """
    records: list[dict[str, Any]] = []
    for cfg in model_configs:
        name = cfg["model_name"]
        model = cfg.get("model")
        if model is None:
            model = get_model(
                name,
                in_channels=int(cfg.get("in_channels", 3)),
                num_classes=int(cfg.get("num_classes", 1)),
                img_size=int(cfg.get("img_size", 256)),
                pretrained=bool(cfg.get("pretrained", True)),
                allow_random_fallback=bool(cfg.get("allow_random_fallback", True)),
                backbone_name=cfg.get("backbone_name"),
            )

        rec = evaluate_model(
            model=model,
            data_loader=data_loader,
            device=device,
            threshold=threshold,
            checkpoint_path=cfg.get("checkpoint_path"),
            dataset_name=dataset_name,
            split=split,
            training_duration_sec=cfg.get("training_duration_sec"),
            is_synthetic=is_synthetic,
        )
        records.append(rec)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    benchmark_id = f"benchmark_{dataset_name.lower().replace(' ', '_')}_{stamp}"
    payload = {
        "benchmark_id": benchmark_id,
        "dataset_name": dataset_name,
        "split": split,
        "threshold": threshold,
        "is_synthetic": is_synthetic,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "models": records,
    }

    if not is_synthetic:
        res_dir = Path(experiments_dir) / "results"
        res_dir.mkdir(parents=True, exist_ok=True)
        (res_dir / f"{benchmark_id}.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        _write_benchmark_csv(records, res_dir / f"{benchmark_id}.csv")
        _write_benchmark_markdown(payload, res_dir / f"{benchmark_id}.md")

    return payload


def _write_benchmark_csv(records: list[dict[str, Any]], out_path: Path) -> None:
    if not records:
        return
    rows = []
    for r in records:
        m = r["metrics"]
        rows.append(
            {
                "model_name": r["model_name"],
                "backbone_name": r["backbone_name"],
                "training_status": r["training_status"],
                "dataset_name": r["dataset_name"],
                "split": r["split"],
                "accuracy": m["accuracy"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1_score": m["f1_score"],
                "iou": m["iou"],
                "dice": m["dice"],
                "total_parameters": r["total_parameters"],
                "checkpoint_size_mb": r["checkpoint_size_mb"],
                "training_duration_sec": r["training_duration_sec"],
                "inference_ms_per_sample": r["inference_ms_per_sample"],
                "peak_memory_mb": r["peak_memory_mb"],
            }
        )
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_benchmark_markdown(payload: dict[str, Any], out_path: Path) -> None:
    lines = [
        f"# Benchmark Report: {payload['dataset_name']} ({payload['split']} split)",
        "",
        f"- **Benchmark ID:** `{payload['benchmark_id']}`",
        f"- **Threshold:** `{payload['threshold']}`",
        f"- **Timestamp (UTC):** {payload['created_at']}",
        "",
        "| Model | Status | Accuracy | Precision | Recall | F1 | IoU | Params | Inference (ms/img) |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in payload["models"]:
        m = r["metrics"]
        params = f"{r['total_parameters']:,}" if r["total_parameters"] else "N/A"
        lines.append(
            f"| {r['display_name']} | `{r['training_status']}` | "
            f"{m['accuracy']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | "
            f"{m['f1_score']:.4f} | {m['iou']:.4f} | {params} | {r['inference_ms_per_sample']:.2f} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_experiment_summaries(
    experiments_dir: str | Path = "experiments",
    include_synthetic: bool = False,
) -> list[dict[str, Any]]:
    """Load real training and benchmark summaries from ``experiments/results/``."""
    results_dir = Path(experiments_dir) / "results"
    if not results_dir.is_dir():
        return []

    records: list[dict[str, Any]] = []
    for json_file in sorted(results_dir.glob("*.json")):
        if json_file.name.endswith("_history.json"):
            continue
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not include_synthetic and data.get("is_synthetic", False):
            continue
        records.append({"file": str(json_file), **data})
    return records


def build_comparison_table(
    experiments_dir: str | Path = "experiments",
    include_synthetic: bool = False,
) -> list[dict[str, Any]]:
    """Build a 3-model comparison table from actual logged results or mark Not evaluated."""
    summaries = load_experiment_summaries(
        experiments_dir=experiments_dir, include_synthetic=include_synthetic
    )
    latest_by_model: dict[str, dict[str, Any]] = {}

    for item in summaries:
        if "models" in item and isinstance(item["models"], list):
            for rec in item["models"]:
                m_key = rec.get("model_name")
                if m_key:
                    latest_by_model[m_key] = rec
        elif "model_name" in item:
            m_key = item["model_name"]
            val_m = item.get("best_val_metrics") or {}
            latest_by_model[m_key] = {
                "model_name": m_key,
                "display_name": MODEL_REGISTRY.get(m_key, {}).get("display_name", m_key),
                "backbone_name": item.get("backbone_name", "unknown"),
                "training_status": item.get("training_status", "unknown"),
                "dataset_name": item.get("dataset_name", "unknown"),
                "split": "val",
                "metrics": val_m,
                "total_parameters": item.get("total_parameters"),
                "checkpoint_size_mb": item.get("checkpoint_size_mb"),
                "training_duration_sec": item.get("training_duration_sec"),
                "inference_ms_per_sample": None,
                "peak_memory_mb": item.get("peak_memory_mb"),
            }

    table: list[dict[str, Any]] = []
    for model_id, meta in MODEL_REGISTRY.items():
        rec = latest_by_model.get(model_id)
        if rec is None:
            table.append(
                {
                    "model_name": model_id,
                    "display_name": meta["display_name"],
                    "status": "Not evaluated",
                    "dataset": "Not evaluated",
                    "accuracy": "Not evaluated",
                    "precision": "Not evaluated",
                    "recall": "Not evaluated",
                    "f1_score": "Not evaluated",
                    "iou": "Not evaluated",
                    "dice": "Not evaluated",
                    "training_duration_sec": "Not evaluated",
                    "inference_ms_per_sample": "Not evaluated",
                    "total_parameters": "Not evaluated",
                    "checkpoint_size_mb": "Not evaluated",
                }
            )
        else:
            m = rec.get("metrics") or {}
            table.append(
                {
                    "model_name": model_id,
                    "display_name": rec.get("display_name", meta["display_name"]),
                    "status": rec.get("training_status", "Not evaluated"),
                    "dataset": f"{rec.get('dataset_name', 'unknown')} ({rec.get('split', 'eval')})",
                    "accuracy": m.get("accuracy", "Not evaluated"),
                    "precision": m.get("precision", "Not evaluated"),
                    "recall": m.get("recall", "Not evaluated"),
                    "f1_score": m.get("f1_score", "Not evaluated"),
                    "iou": m.get("iou", "Not evaluated"),
                    "dice": m.get("dice", "Not evaluated"),
                    "training_duration_sec": rec.get("training_duration_sec") or "Not evaluated",
                    "inference_ms_per_sample": rec.get("inference_ms_per_sample") or "Not evaluated",
                    "total_parameters": rec.get("total_parameters") or "Not evaluated",
                    "checkpoint_size_mb": rec.get("checkpoint_size_mb") or "Not evaluated",
                }
            )
    return table

