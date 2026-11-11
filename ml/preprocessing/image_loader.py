"""Unified satellite and RGB image loader with validation and geospatial metadata preservation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import rasterio
from PIL import Image, UnidentifiedImageError
from rasterio.enums import Resampling
from rasterio.warp import reproject

SUPPORTED_IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
)


class ImageValidationError(ValueError):
    """Raised when an uploaded or dataset image fails format, integrity, or compatibility checks."""


@dataclass
class GeoMetadata:
    """Geospatial and structural raster metadata preserved across preprocessing."""

    width: int
    height: int
    count: int
    dtype: str
    crs: str | None = None
    transform: tuple[float, float, float, float, float, float] | None = None
    bounds: tuple[float, float, float, float] | None = None
    nodata: float | None = None
    band_names: list[str] | None = None
    driver: str | None = None
    is_georeferenced: bool = False
    source_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize metadata into a JSON-compatible dictionary."""
        return asdict(self)


@dataclass
class LoadedImage:
    """In-memory normalized image container with valid-data mask and metadata.

    Attributes:
        data: Float32 array of shape ``(C, H, W)``.
        valid_mask: Boolean array of shape ``(H, W)`` where True marks valid pixels.
        metadata: Geospatial and raster metadata.
        raw_dtype: Original file dtype string.
        has_spectral_quality_bands: True if multispectral/Sentinel-2 quality bands are present.
        warnings: Non-fatal informational notes collected during loading.
    """

    data: np.ndarray
    valid_mask: np.ndarray
    metadata: GeoMetadata
    raw_dtype: str
    has_spectral_quality_bands: bool = False
    warnings: list[str] = field(default_factory=list)

    @property
    def channels(self) -> int:
        return int(self.data.shape[0])

    @property
    def height(self) -> int:
        return int(self.data.shape[1])

    @property
    def width(self) -> int:
        return int(self.data.shape[2])

    @property
    def shape_chw(self) -> tuple[int, int, int]:
        return (self.channels, self.height, self.width)

    def to_hwc(self) -> np.ndarray:
        """Return array in ``(H, W, C)`` channel-last layout."""
        return np.transpose(self.data, (1, 2, 0))

    def to_rgb_uint8(self, rgb_indices: tuple[int, int, int] = (0, 1, 2)) -> np.ndarray:
        """Convert image (or selected 3 bands) to ``(H, W, 3)`` uint8 for display."""
        c, h, w = self.data.shape
        if c == 1:
            rgb = np.repeat(self.data, 3, axis=0)
        elif c >= 3:
            idx = [min(max(i, 0), c - 1) for i in rgb_indices]
            rgb = self.data[idx]
        else:
            # 2-band edge case: pad third channel with second band
            rgb = np.stack([self.data[0], self.data[1], self.data[1]], axis=0)

        arr = np.transpose(rgb, (1, 2, 0)).astype(np.float32)
        finite = np.isfinite(arr)
        if not np.any(finite):
            return np.zeros((h, w, 3), dtype=np.uint8)

        max_val = float(np.nanmax(arr[finite]))
        min_val = float(np.nanmin(arr[finite]))
        if max_val <= 1.0 and min_val >= 0.0:
            scaled = arr * 255.0
        elif max_val <= 255.0 and min_val >= 0.0:
            scaled = arr
        else:
            # Percentile contrast stretch over valid pixels for high-bit-depth imagery (e.g. Sentinel-2 uint16)
            scaled = np.zeros_like(arr, dtype=np.float32)
            for ch in range(3):
                ch_vals = arr[:, :, ch][self.valid_mask & np.isfinite(arr[:, :, ch])]
                if ch_vals.size == 0:
                    continue
                lo = float(np.percentile(ch_vals, 2))
                hi = float(np.percentile(ch_vals, 98))
                denom = max(hi - lo, 1e-6)
                scaled[:, :, ch] = np.clip((arr[:, :, ch] - lo) / denom, 0.0, 1.0) * 255.0
        scaled = np.nan_to_num(scaled, nan=0.0, posinf=255.0, neginf=0.0)
        return np.clip(np.round(scaled), 0, 255).astype(np.uint8)


@dataclass
class PairValidationResult:
    """Validation summary for a before-and-after image pair."""

    is_valid: bool
    same_dimensions: bool
    same_channels: bool
    both_georeferenced: bool
    crs_match: bool | None
    transform_match: bool | None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def validate_image_file(
    path: str | Path,
    max_size_mb: float = 50.0,
    min_dim: int = 16,
    max_dim: int = 20000,
) -> Path:
    """Validate file existence, extension, size bounds, and non-emptiness."""
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise ImageValidationError(f"Image file does not exist: {p}")

    ext = p.suffix.lower()
    if ext not in SUPPORTED_IMAGE_EXTENSIONS:
        raise ImageValidationError(
            f"Unsupported image format '{ext}' for '{p.name}'. "
            f"Supported formats: {sorted(SUPPORTED_IMAGE_EXTENSIONS)}"
        )

    size_bytes = p.stat().st_size
    if size_bytes == 0:
        raise ImageValidationError(f"Image file is empty (0 bytes): {p.name}")

    max_bytes = int(max_size_mb * 1024 * 1024)
    if size_bytes > max_bytes:
        raise ImageValidationError(
            f"Image '{p.name}' ({size_bytes / (1024 * 1024):.2f} MB) exceeds "
            f"maximum allowed size of {max_size_mb:.1f} MB."
        )

    if min_dim < 1 or max_dim < min_dim:
        raise ValueError("Invalid min_dim/max_dim constraints.")

    return p


def _load_tiff_with_rasterio(path: Path) -> LoadedImage:
    """Load TIFF or GeoTIFF preserving CRS, affine transform, nodata, and band names."""
    try:
        with rasterio.open(path) as src:
            width = int(src.width)
            height = int(src.height)
            count = int(src.count)
            if width == 0 or height == 0 or count == 0:
                raise ImageValidationError(f"TIFF '{path.name}' has zero dimensions or bands.")

            arr = src.read().astype(np.float32)  # (C, H, W)
            raw_dtype = str(src.dtypes[0]) if src.dtypes else str(arr.dtype)
            nodata = float(src.nodata) if src.nodata is not None else None

            # Rasterio dataset_mask returns 0 where nodata/mask applies, 255 where valid
            ds_mask = src.dataset_mask()
            valid_mask = (ds_mask > 0) & np.all(np.isfinite(arr), axis=0)
            if nodata is not None and not np.isnan(nodata):
                valid_mask &= ~np.all(np.isclose(arr, nodata), axis=0)

            crs_str = src.crs.to_string() if src.crs is not None else None
            has_non_identity_transform = src.crs is not None
            transform_tuple = (
                tuple(float(x) for x in src.transform[:6]) if src.transform is not None else None
            )
            bounds_tuple = (
                tuple(float(x) for x in src.bounds) if has_non_identity_transform else None
            )
            band_names = [
                desc if desc else f"Band_{i + 1}"
                for i, desc in enumerate(src.descriptions or [None] * count)
            ]

            meta = GeoMetadata(
                width=width,
                height=height,
                count=count,
                dtype=raw_dtype,
                crs=crs_str,
                transform=transform_tuple,
                bounds=bounds_tuple,
                nodata=nodata,
                band_names=band_names,
                driver=src.driver,
                is_georeferenced=bool(src.crs is not None),
                source_path=str(path),
            )
            has_quality = any(
                b.upper() in {"SCL", "MSK_CLDPRB", "QA60"} for b in band_names
            ) or count >= 4
            warnings: list[str] = []
            if not meta.is_georeferenced:
                warnings.append(
                    f"TIFF '{path.name}' has no CRS metadata; area statistics will be image-relative."
                )
            return LoadedImage(
                data=arr,
                valid_mask=valid_mask,
                metadata=meta,
                raw_dtype=raw_dtype,
                has_spectral_quality_bands=has_quality,
                warnings=warnings,
            )
    except rasterio.errors.RasterioIOError as exc:
        raise ImageValidationError(f"Corrupted or unreadable TIFF '{path.name}': {exc}") from exc


def _load_standard_with_pil(path: Path) -> LoadedImage:
    """Load PNG/JPG/JPEG via Pillow with corruption verification."""
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            mode = img.mode
            if mode == "RGBA":
                rgba = np.asarray(img, dtype=np.float32)
                alpha = rgba[:, :, 3]
                valid_mask = (alpha > 0) & np.all(np.isfinite(rgba[:, :, :3]), axis=2)
                arr_hwc = rgba[:, :, :3]
            elif mode in {"L", "I;16", "I", "F"}:
                gray = np.asarray(img, dtype=np.float32)
                valid_mask = np.isfinite(gray)
                arr_hwc = gray[:, :, None]
            else:
                rgb = img.convert("RGB")
                arr_hwc = np.asarray(rgb, dtype=np.float32)
                valid_mask = np.all(np.isfinite(arr_hwc), axis=2)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageValidationError(
            f"Corrupted or unreadable image file '{path.name}': {exc}"
        ) from exc

    h, w, c = arr_hwc.shape
    arr_chw = np.transpose(arr_hwc, (2, 0, 1)).astype(np.float32)
    meta = GeoMetadata(
        width=w,
        height=h,
        count=c,
        dtype="uint8",
        crs=None,
        transform=None,
        bounds=None,
        nodata=None,
        band_names=[f"Channel_{i + 1}" for i in range(c)],
        driver=path.suffix.lstrip(".").upper(),
        is_georeferenced=False,
        source_path=str(path),
    )
    warnings = [
        "Standard RGB/grayscale upload lacks geospatial CRS and Sentinel-2 quality bands; "
        "geographic area and reliable satellite cloud masking are unavailable."
    ]
    return LoadedImage(
        data=arr_chw,
        valid_mask=valid_mask,
        metadata=meta,
        raw_dtype="uint8",
        has_spectral_quality_bands=False,
        warnings=warnings,
    )


def load_image(
    path: str | Path,
    expected_channels: int | None = None,
    max_size_mb: float = 50.0,
    min_dim: int = 16,
    max_dim: int = 20000,
) -> LoadedImage:
    """Load and validate an image (PNG, JPG, JPEG, TIFF, GeoTIFF)."""
    p = validate_image_file(path, max_size_mb=max_size_mb, min_dim=min_dim, max_dim=max_dim)
    ext = p.suffix.lower()

    if ext in {".tif", ".tiff"}:
        loaded = _load_tiff_with_rasterio(p)
    else:
        loaded = _load_standard_with_pil(p)

    if loaded.width < min_dim or loaded.height < min_dim:
        raise ImageValidationError(
            f"Image '{p.name}' dimensions ({loaded.width}x{loaded.height}) are smaller "
            f"than minimum required {min_dim}x{min_dim}."
        )
    if loaded.width > max_dim or loaded.height > max_dim:
        raise ImageValidationError(
            f"Image '{p.name}' dimensions ({loaded.width}x{loaded.height}) exceed "
            f"maximum supported {max_dim}x{max_dim}."
        )

    if expected_channels is not None and loaded.channels != expected_channels:
        if expected_channels == 3 and loaded.channels == 1:
            loaded.data = np.repeat(loaded.data, 3, axis=0)
            loaded.metadata.count = 3
            loaded.metadata.band_names = ["Gray_R", "Gray_G", "Gray_B"]
            loaded.warnings.append(
                f"Expanded single-channel image '{p.name}' to 3 channels for model compatibility."
            )
        elif expected_channels == 3 and loaded.channels > 3:
            raise ImageValidationError(
                f"Image '{p.name}' has {loaded.channels} channels, expected {expected_channels}. "
                "Select RGB bands explicitly before RGB-only inference."
            )
        else:
            raise ImageValidationError(
                f"Image '{p.name}' has {loaded.channels} channel(s), expected {expected_channels}."
            )

    return loaded


def load_mask(path: str | Path, max_size_mb: float = 50.0) -> np.ndarray:
    """Load a binary ground-truth change mask as ``(H, W)`` uint8 values in ``{0, 1}``."""
    p = validate_image_file(path, max_size_mb=max_size_mb, min_dim=1, max_dim=20000)
    if p.suffix.lower() in {".tif", ".tiff"}:
        with rasterio.open(p) as src:
            arr = src.read(1)
    else:
        try:
            with Image.open(p) as img:
                arr = np.asarray(img.convert("L"))
        except (UnidentifiedImageError, OSError) as exc:
            raise ImageValidationError(f"Corrupted mask file '{p.name}': {exc}") from exc

    unique_vals = set(np.unique(arr))
    # Handle OSCD encoding: 1 = no change, 2 = change (0 = nodata)
    if 2 in unique_vals and unique_vals.issubset({0, 1, 2}):
        return (arr == 2).astype(np.uint8)
    if 255 in unique_vals:
        return (arr == 255).astype(np.uint8)
    return (arr > 0).astype(np.uint8)


def load_multispectral_stack(
    band_paths: Sequence[str | Path],
    band_names: Sequence[str] | None = None,
    reference_index: int = 0,
    max_size_mb: float = 100.0,
) -> LoadedImage:
    """Stack single-band GeoTIFF files onto a shared reference grid.

    Useful for Sentinel-2 / OSCD datasets where bands have 10m, 20m, and 60m
    native spatial resolutions. Lower-resolution bands are bilinearly resampled
    to the reference band's grid (usually a 10m band such as B02/B04/B08).
    """
    if not band_paths:
        raise ImageValidationError("No band paths provided to load_multispectral_stack.")

    paths = [validate_image_file(bp, max_size_mb=max_size_mb, min_dim=1) for bp in band_paths]
    ref_path = paths[reference_index]

    with rasterio.open(ref_path) as ref_src:
        ref_height = int(ref_src.height)
        ref_width = int(ref_src.width)
        ref_crs = ref_src.crs
        ref_transform = ref_src.transform
        ref_bounds = tuple(float(x) for x in ref_src.bounds) if ref_crs is not None else None
        ref_nodata = float(ref_src.nodata) if ref_src.nodata is not None else None
        ref_dtype = str(ref_src.dtypes[0]) if ref_src.dtypes else "float32"

    stacked = np.zeros((len(paths), ref_height, ref_width), dtype=np.float32)
    valid_mask = np.ones((ref_height, ref_width), dtype=bool)
    resolved_names: list[str] = []
    warnings: list[str] = []

    for idx, bp in enumerate(paths):
        bname = band_names[idx] if band_names and idx < len(band_names) else bp.stem
        resolved_names.append(bname)
        with rasterio.open(bp) as src:
            band_arr = src.read(1).astype(np.float32)
            same_grid = (
                src.height == ref_height
                and src.width == ref_width
                and src.crs == ref_crs
                and src.transform == ref_transform
            )
            if same_grid:
                resampled = band_arr
            elif src.crs is not None and ref_crs is not None:
                resampled = np.zeros((ref_height, ref_width), dtype=np.float32)
                reproject(
                    source=band_arr,
                    destination=resampled,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=ref_transform,
                    dst_crs=ref_crs,
                    resampling=Resampling.bilinear,
                )
                warnings.append(
                    f"Resampled band {bname} from {src.width}x{src.height} to {ref_width}x{ref_height}."
                )
            else:
                import cv2

                resampled = cv2.resize(
                    band_arr, (ref_width, ref_height), interpolation=cv2.INTER_LINEAR
                )
                warnings.append(
                    f"Resized non-georeferenced band {bname} to {ref_width}x{ref_height}."
                )

            stacked[idx] = resampled
            valid_mask &= np.isfinite(resampled)
            if src.nodata is not None and not np.isnan(src.nodata):
                valid_mask &= ~np.isclose(resampled, float(src.nodata))

    meta = GeoMetadata(
        width=ref_width,
        height=ref_height,
        count=len(paths),
        dtype=ref_dtype,
        crs=ref_crs.to_string() if ref_crs is not None else None,
        transform=tuple(float(x) for x in ref_transform[:6]) if ref_transform is not None else None,
        bounds=ref_bounds,
        nodata=ref_nodata,
        band_names=resolved_names,
        driver="GTiff",
        is_georeferenced=bool(ref_crs is not None),
        source_path=str(ref_path.parent),
    )
    has_quality = any(n.upper() in {"SCL", "MSK_CLDPRB", "QA60"} for n in resolved_names) or len(
        paths
    ) >= 4
    return LoadedImage(
        data=stacked,
        valid_mask=valid_mask,
        metadata=meta,
        raw_dtype=ref_dtype,
        has_spectral_quality_bands=has_quality,
        warnings=warnings,
    )


def validate_image_pair(
    before: LoadedImage,
    after: LoadedImage,
    require_same_dimensions: bool = False,
    require_same_crs: bool = False,
) -> PairValidationResult:
    """Validate channel, spatial, and geospatial compatibility of an image pair."""
    errors: list[str] = []
    warnings: list[str] = list(before.warnings) + list(after.warnings)

    same_channels = before.channels == after.channels
    if not same_channels:
        errors.append(
            f"Incompatible channel count: before has {before.channels} channel(s), "
            f"after has {after.channels} channel(s)."
        )

    same_dims = before.height == after.height and before.width == after.width
    if not same_dims:
        msg = (
            f"Image dimensions differ: before is {before.width}x{before.height}, "
            f"after is {after.width}x{after.height}."
        )
        if require_same_dimensions:
            errors.append(msg)
        else:
            warnings.append(msg + " Alignment/resampling will be required before inference.")

    both_geo = before.metadata.is_georeferenced and after.metadata.is_georeferenced
    crs_match: bool | None = None
    transform_match: bool | None = None

    if both_geo:
        crs_match = before.metadata.crs == after.metadata.crs
        transform_match = before.metadata.transform == after.metadata.transform
        if not crs_match:
            msg = (
                f"CRS mismatch between temporal images: before={before.metadata.crs}, "
                f"after={after.metadata.crs}."
            )
            if require_same_crs:
                errors.append(msg)
            else:
                warnings.append(msg + " Reprojection to a common CRS is required.")
        elif not transform_match:
            warnings.append(
                "Affine geotransforms differ despite matching CRS; reprojection to a common grid will be applied."
            )
    elif before.metadata.is_georeferenced != after.metadata.is_georeferenced:
        warnings.append(
            "Only one image carries georeferencing metadata; geographic alignment cannot be guaranteed."
        )

    overlap_valid = bool(np.any(before.valid_mask)) and bool(np.any(after.valid_mask))
    if not overlap_valid:
        errors.append("One or both images contain zero valid pixels.")

    return PairValidationResult(
        is_valid=len(errors) == 0,
        same_dimensions=same_dims,
        same_channels=same_channels,
        both_georeferenced=both_geo,
        crs_match=crs_match,
        transform_match=transform_match,
        errors=errors,
        warnings=warnings,
    )

