"""Preprocessing pipeline exports."""

from ml.preprocessing.alignment import AlignmentReport, align_image_pair, apply_ecc_registration
from ml.preprocessing.cloud_masking import (
    DEFAULT_SCL_MASK_CLASSES,
    CloudMaskConfig,
    CloudMaskReport,
    apply_cloud_mask,
)
from ml.preprocessing.image_loader import (
    SUPPORTED_IMAGE_EXTENSIONS,
    GeoMetadata,
    ImageValidationError,
    LoadedImage,
    PairValidationResult,
    load_image,
    load_mask,
    load_multispectral_stack,
    validate_image_file,
    validate_image_pair,
)
from ml.preprocessing.normalization import normalize_image, normalize_pair
from ml.preprocessing.tiling import (
    TileSpec,
    compute_tile_grid,
    create_blend_window,
    extract_patches,
    pad_chw_to_min_size,
    stitch_tiles,
)

__all__ = [
    "AlignmentReport",
    "CloudMaskConfig",
    "CloudMaskReport",
    "DEFAULT_SCL_MASK_CLASSES",
    "GeoMetadata",
    "ImageValidationError",
    "LoadedImage",
    "PairValidationResult",
    "SUPPORTED_IMAGE_EXTENSIONS",
    "TileSpec",
    "align_image_pair",
    "apply_cloud_mask",
    "apply_ecc_registration",
    "compute_tile_grid",
    "create_blend_window",
    "extract_patches",
    "load_image",
    "load_mask",
    "load_multispectral_stack",
    "normalize_image",
    "normalize_pair",
    "pad_chw_to_min_size",
    "stitch_tiles",
    "validate_image_file",
    "validate_image_pair",
]

