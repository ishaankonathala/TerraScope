"""Lazy model loading, LRU memory caching, and safe checkpoint inspection."""

from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any

from backend.app.core.config import Settings
from backend.app.utils.files import safe_resolve_child
from ml.models import MODEL_REGISTRY, BaseChangeDetector, canonical_model_name, get_model, list_available_models
from ml.training.callbacks import load_checkpoint
from ml.utils.device import detect_device

logger = logging.getLogger(__name__)


class ModelService:
    """Loads models lazily on demand with an LRU cache to protect Apple M1 RAM."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._cache: OrderedDict[tuple[str, str | None, int], BaseChangeDetector] = OrderedDict()

    def list_checkpoints(self) -> list[dict[str, Any]]:
        ckpt_dir = self.settings.models_path
        if not ckpt_dir.is_dir():
            return []
        items: list[dict[str, Any]] = []
        for path in sorted(ckpt_dir.glob("*.pt")):
            items.append(
                {
                    "filename": path.name,
                    "size_mb": round(path.stat().st_size / (1024.0 * 1024.0), 3),
                }
            )
        return items

    def get_models_catalog(self) -> dict[str, Any]:
        ckpts = self.list_checkpoints()
        catalog = []
        for entry in list_available_models():
            mid = entry["model_id"]
            matching_ckpts = [c for c in ckpts if c["filename"].startswith(mid)]
            catalog.append(
                {
                    **entry,
                    "available_checkpoints": matching_ckpts,
                    "has_trained_checkpoint": len(matching_ckpts) > 0,
                    "availability_note": (
                        "Trained checkpoint available."
                        if matching_ckpts
                        else "No fine-tuned checkpoint found; inference uses backbone weights/untrained head until trained."
                    ),
                }
            )
        return {"models": catalog, "checkpoints": ckpts}

    def get_or_load_model(
        self,
        model_name: str,
        checkpoint_name: str | None = None,
        in_channels: int = 3,
        pretrained: bool | None = None,
    ) -> BaseChangeDetector:
        canon = canonical_model_name(model_name)
        key = (canon, checkpoint_name, in_channels)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        use_pretrained = (
            self.settings.allow_pretrained_download if pretrained is None else pretrained
        )
        # If a full checkpoint exists, no need to download pretrained weights from internet first
        if checkpoint_name is not None:
            use_pretrained = False

        model = get_model(
            canon,
            in_channels=in_channels,
            num_classes=1,
            img_size=self.settings.default_image_size,
            pretrained=use_pretrained,
            allow_random_fallback=True,
        )

        if checkpoint_name:
            ckpt_path = safe_resolve_child(self.settings.models_path, Path(checkpoint_name).name)
            load_checkpoint(ckpt_path, model, map_location="cpu")

        dev = detect_device(self.settings.device)
        model.to(dev)
        model.eval()

        while len(self._cache) >= max(self.settings.max_cached_models, 1):
            evicted_key, _ = self._cache.popitem(last=False)
            logger.info("Evicted model %s from LRU cache to conserve memory.", evicted_key)

        self._cache[key] = model
        return model

