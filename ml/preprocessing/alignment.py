"""Temporal spatial alignment, georeferenced reprojection, and optional ECC registration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np
from affine import Affine
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.warp import reproject

from ml.preprocessing.image_loader import GeoMetadata, ImageValidationError, LoadedImage, validate_image_pair


@dataclass
class AlignmentReport:
    """Audit record documenting how a temporal image pair was aligned."""

    method: str
    reprojected: bool
    resized: bool
    ecc_applied: bool
    before_shape: tuple[int, int]
    after_shape_original: tuple[int, int]
    output_shape: tuple[int, int]
    crs: str | None
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _resize_chw(
    data: np.ndarray,
    valid_mask: np.ndarray,
    target_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Resize ``(C, H, W)`` float32 array and ``(H, W)`` valid mask cleanly."""
    target_h, target_w = target_hw
    c, h, w = data.shape
    if h == target_h and w == target_w:
        return data, valid_mask

    hwc = np.transpose(data, (1, 2, 0))
    resized_hwc = cv2.resize(hwc, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    if resized_hwc.ndim == 2:
        resized_hwc = resized_hwc[:, :, None]
    resized_chw = np.transpose(resized_hwc, (2, 0, 1)).astype(np.float32)

    resized_mask = cv2.resize(
        valid_mask.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST
    )
    return resized_chw, resized_mask > 0


def _reproject_to_reference_grid(
    source: LoadedImage,
    ref_meta: GeoMetadata,
    resampling: Resampling = Resampling.bilinear,
) -> LoadedImage:
    """Reproject a georeferenced LoadedImage onto the reference GeoMetadata grid."""
    if not source.metadata.is_georeferenced or not ref_meta.is_georeferenced:
        raise ImageValidationError("Both images must be georeferenced for raster reprojection.")
    if source.metadata.transform is None or ref_meta.transform is None:
        raise ImageValidationError("Missing affine transform required for raster reprojection.")

    src_crs = CRS.from_user_input(source.metadata.crs)
    dst_crs = CRS.from_user_input(ref_meta.crs)
    src_transform = Affine(*source.metadata.transform)
    dst_transform = Affine(*ref_meta.transform)

    c = source.channels
    dst_h, dst_w = ref_meta.height, ref_meta.width
    dst_data = np.zeros((c, dst_h, dst_w), dtype=np.float32)

    for band_idx in range(c):
        reproject(
            source=source.data[band_idx],
            destination=dst_data[band_idx],
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            src_nodata=source.metadata.nodata,
            dst_nodata=np.nan,
            resampling=resampling,
        )

    dst_valid_u8 = np.zeros((dst_h, dst_w), dtype=np.uint8)
    reproject(
        source=source.valid_mask.astype(np.uint8),
        destination=dst_valid_u8,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        src_nodata=0,
        dst_nodata=0,
        resampling=Resampling.nearest,
    )
    dst_valid = (dst_valid_u8 > 0) & np.all(np.isfinite(dst_data), axis=0)
    dst_data = np.nan_to_num(dst_data, nan=0.0)

    new_meta = GeoMetadata(
        width=dst_w,
        height=dst_h,
        count=c,
        dtype=source.metadata.dtype,
        crs=ref_meta.crs,
        transform=ref_meta.transform,
        bounds=ref_meta.bounds,
        nodata=source.metadata.nodata,
        band_names=source.metadata.band_names,
        driver=source.metadata.driver,
        is_georeferenced=True,
        source_path=source.metadata.source_path,
    )
    return LoadedImage(
        data=dst_data,
        valid_mask=dst_valid,
        metadata=new_meta,
        raw_dtype=source.raw_dtype,
        has_spectral_quality_bands=source.has_spectral_quality_bands,
        warnings=list(source.warnings),
    )


def apply_ecc_registration(
    before: LoadedImage,
    after: LoadedImage,
    motion_mode: str = "affine",
    max_iterations: int = 100,
) -> tuple[LoadedImage, bool]:
    """Attempt intensity-based OpenCV ECC co-registration on same-size temporal images.

    Useful for studying misalignment sensitivity (RQ3) or correcting minor pixel drift
    on non-georeferenced imagery. Returns ``(aligned_after, converged)``.
    """
    if before.shape_chw != after.shape_chw:
        raise ImageValidationError("ECC registration requires identical shapes before/after.")

    ref_gray = np.mean(before.to_rgb_uint8(), axis=2).astype(np.float32) / 255.0
    mov_gray = np.mean(after.to_rgb_uint8(), axis=2).astype(np.float32) / 255.0

    if motion_mode == "translation":
        warp_mode = cv2.MOTION_TRANSLATION
        warp_matrix = np.eye(2, 3, dtype=np.float32)
    else:
        warp_mode = cv2.MOTION_AFFINE
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iterations, 1e-5)
    try:
        _, warp_matrix = cv2.findTransformECC(ref_gray, mov_gray, warp_matrix, warp_mode, criteria)
    except cv2.error:
        return after, False

    h, w = before.height, before.width
    hwc = after.to_hwc()
    warped_hwc = cv2.warpAffine(
        hwc,
        warp_matrix,
        (w, h),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    if warped_hwc.ndim == 2:
        warped_hwc = warped_hwc[:, :, None]
    warped_mask = cv2.warpAffine(
        after.valid_mask.astype(np.uint8),
        warp_matrix,
        (w, h),
        flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    aligned = LoadedImage(
        data=np.transpose(warped_hwc, (2, 0, 1)).astype(np.float32),
        valid_mask=warped_mask > 0,
        metadata=after.metadata,
        raw_dtype=after.raw_dtype,
        has_spectral_quality_bands=after.has_spectral_quality_bands,
        warnings=list(after.warnings) + ["Applied OpenCV ECC registration to after image."],
    )
    return aligned, True


def align_image_pair(
    before: LoadedImage,
    after: LoadedImage,
    target_size: tuple[int, int] | None = None,
    use_ecc: bool = False,
) -> tuple[LoadedImage, LoadedImage, AlignmentReport]:
    """Align a before/after image pair onto a common spatial and geospatial grid.

    - If both images are georeferenced and differ in CRS, transform, or grid size,
      reprojects ``after`` onto ``before``'s exact CRS and affine grid (never
      assuming matching pixel dimensions implies geographic alignment).
    - If ``target_size=(H, W)`` is supplied, resizes both images consistently
      (updating affine transform scale if georeferenced).
    """
    validation = validate_image_pair(before, after)
    if not validation.is_valid:
        raise ImageValidationError("; ".join(validation.errors))

    warnings = list(validation.warnings)
    orig_before_hw = (before.height, before.width)
    orig_after_hw = (after.height, after.width)
    reprojected = False
    resized = False
    ecc_applied = False

    if validation.both_georeferenced:
        grid_matches = (
            bool(validation.crs_match)
            and bool(validation.transform_match)
            and validation.same_dimensions
        )
        if not grid_matches:
            after = _reproject_to_reference_grid(after, before.metadata)
            reprojected = True
            method = "georeferenced_reprojected"
        else:
            method = "georeferenced_exact"
    else:
        if not validation.same_dimensions:
            resized_data, resized_mask = _resize_chw(after.data, after.valid_mask, orig_before_hw)
            after_meta = GeoMetadata(
                width=orig_before_hw[1],
                height=orig_before_hw[0],
                count=after.channels,
                dtype=after.raw_dtype,
                crs=after.metadata.crs,
                transform=after.metadata.transform,
                bounds=after.metadata.bounds,
                nodata=after.metadata.nodata,
                band_names=after.metadata.band_names,
                driver=after.metadata.driver,
                is_georeferenced=after.metadata.is_georeferenced,
                source_path=after.metadata.source_path,
            )
            after = LoadedImage(
                data=resized_data,
                valid_mask=resized_mask,
                metadata=after_meta,
                raw_dtype=after.raw_dtype,
                has_spectral_quality_bands=after.has_spectral_quality_bands,
                warnings=list(after.warnings),
            )
            resized = True
            method = "non_georeferenced_resized"
        else:
            method = "pixel_dimensions_match"

    if use_ecc:
        after, converged = apply_ecc_registration(before, after)
        ecc_applied = converged
        if converged:
            method += "+ecc"
        else:
            warnings.append("ECC registration did not converge; kept initial alignment.")

    if target_size is not None and (before.height, before.width) != target_size:
        th, tw = target_size
        b_data, b_mask = _resize_chw(before.data, before.valid_mask, target_size)
        a_data, a_mask = _resize_chw(after.data, after.valid_mask, target_size)

        def _scaled_meta(meta: GeoMetadata) -> GeoMetadata:
            new_tf = meta.transform
            if meta.is_georeferenced and meta.transform is not None:
                sx = meta.width / float(tw)
                sy = meta.height / float(th)
                a, b, c, d, e, f = meta.transform
                new_tf = (a * sx, b * sx, c, d * sy, e * sy, f)
            return GeoMetadata(
                width=tw,
                height=th,
                count=meta.count,
                dtype=meta.dtype,
                crs=meta.crs,
                transform=new_tf,
                bounds=meta.bounds,
                nodata=meta.nodata,
                band_names=meta.band_names,
                driver=meta.driver,
                is_georeferenced=meta.is_georeferenced,
                source_path=meta.source_path,
            )

        before = LoadedImage(
            data=b_data,
            valid_mask=b_mask,
            metadata=_scaled_meta(before.metadata),
            raw_dtype=before.raw_dtype,
            has_spectral_quality_bands=before.has_spectral_quality_bands,
            warnings=list(before.warnings),
        )
        after = LoadedImage(
            data=a_data,
            valid_mask=a_mask,
            metadata=_scaled_meta(after.metadata),
            raw_dtype=after.raw_dtype,
            has_spectral_quality_bands=after.has_spectral_quality_bands,
            warnings=list(after.warnings),
        )
        resized = True

    combined_valid = before.valid_mask & after.valid_mask
    if not np.any(combined_valid):
        raise ImageValidationError(
            "Aligned before and after images have zero overlapping valid pixels."
        )

    before.valid_mask = combined_valid
    after.valid_mask = combined_valid

    report = AlignmentReport(
        method=method,
        reprojected=reprojected,
        resized=resized,
        ecc_applied=ecc_applied,
        before_shape=orig_before_hw,
        after_shape_original=orig_after_hw,
        output_shape=(before.height, before.width),
        crs=before.metadata.crs,
        warnings=warnings,
    )
    return before, after, report

