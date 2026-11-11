"""Utility helpers for ML modules."""

from ml.utils.device import (
    DeviceInfo,
    detect_device,
    device_info_dict,
    get_device_info,
    get_peak_memory_mb,
    reset_peak_memory_stats,
    resolve_device_safe,
    set_seed,
)

__all__ = [
    "DeviceInfo",
    "detect_device",
    "device_info_dict",
    "get_device_info",
    "get_peak_memory_mb",
    "reset_peak_memory_stats",
    "resolve_device_safe",
    "set_seed",
]

