# Dataset Documentation

This project supports three dataset workflows for bi-temporal change detection: **LEVIR-CD** (building change detection), **OSCD** (multispectral Sentinel-2 urban change detection), and **Custom User Imagery** (arbitrary paired GeoTIFF/TIFF/PNG/JPG imagery).

No copyrighted or third-party public datasets are redistributed in this repository, and missing datasets are never fabricated or substituted with synthetic images.

---

## 1. Supported Datasets Overview

| Dataset | Purpose | Modalities & Spatial Scale | Label Semantics | Official Source & Licensing |
| :--- | :--- | :--- | :--- | :--- |
| **LEVIR-CD** | Building construction & demolition change detection | High-resolution (~0.5 m) RGB `1024x1024` tiles (`train`, `val`, `test`) | Binary mask (`0` = unchanged, `1` = changed building) | [LEVIR-CD Website](https://justchenhao.github.io/LEVIR/) — check official academic usage terms before downloading |
| **OSCD** *(Onera Satellite Change Detection)* | Multi-temporal urban land-use & built-up change detection | 13 Sentinel-2 bands across 24 cities (`10m` / `20m` / `60m` native GSD) | Binary pixel change mask (`0` = unchanged, `1` = changed) | [OSCD Website](https://rcdaudt.github.io/oscd/) — CC-BY-4.0 / Onera academic distribution terms |
| **Custom Pairs** | User-supplied bi-temporal scenes & Sentinel-2 GeoTIFF exports | 1-band, 3-band RGB, or multi-band `.tif`/`.tiff`/`.png`/`.jpg`/`.jpeg` | Optional binary mask (`0`/`255` or `0`/`1`) | User-provided (preserves GeoTIFF CRS & transform if present) |

> [!NOTE]
> Both **LEVIR-CD** and **OSCD** provide **binary** change masks (`no_change` vs `change`). The evaluation pipeline preserves that distinction and never relabels binary predictions as multi-class land-cover transitions (e.g., deforestation or urbanization) unless multi-class labels or external spectral indices are supplied.

---

## 2. Dataset A: LEVIR-CD

### Directory Layout

Place downloaded data under `data/raw/LEVIR-CD/`:

```text
data/raw/LEVIR-CD/
├── train/
│   ├── A/      # Before RGB images
│   ├── B/      # After RGB images
│   └── label/  # Binary change masks (0 / 255)
├── val/
│   ├── A/
│   ├── B/
│   └── label/
└── test/
    ├── A/
    ├── B/
    └── label/
```

### Loader Implementation

Handled by [`LEVIRCDDataset`](file:///Users/ishaan/satellite/ml/datasets/levir_cd.py#L18):
- Validates split membership (`train`, `val`, `test`) and matching stems across `A/`, `B/`, and `label/`.
- Supports extracting fixed-size patches (`patch_size=256`, `stride=256` by default) from `1024x1024` scenes via [`compute_tile_grid`](file:///Users/ishaan/satellite/ml/preprocessing/tiling.py) or resizing whole images when `patch_size=None`.
- Uses [`PairedChangeTransform`](file:///Users/ishaan/satellite/ml/datasets/transforms.py#L36) so before/after tensors and masks undergo **identical** geometric flips/rotations/crops alongside subtle radiometric jitter on training splits.

---

## 3. Dataset B: OSCD (Onera Satellite Change Detection)

### Directory Layout

Place extracted city folders under `data/raw/OSCD/`:

```text
data/raw/OSCD/
├── images/
│   └── <city_name>/        # e.g., abudhabi, dubai, paris, rennes
│       ├── imgs_1_rect/    # B01.tif ... B12.tif (Date 1)
│       ├── imgs_2_rect/    # B01.tif ... B12.tif (Date 2)
│       └── cm/cm.png       # (or separate labels/<city>/cm/cm.png)
└── labels/                 # Optional external label tree
```

### Official Splits & Multispectral Band Handling

Defined in [`OSCDDataset`](file:///Users/ishaan/satellite/ml/datasets/oscd.py#L70):
- **Official Split Assignment:** Uses the 14 official training cities (`OSCD_TRAIN_CITIES`, holding out `("paris", "pisa", "mumbai")` by default as `val`) and 10 official test cities (`OSCD_TEST_CITIES`).
- **13 Sentinel-2 Bands Supported (`SENTINEL2_13_BANDS`):** `B01`, `B02`, `B03`, `B04`, `B05`, `B06`, `B07`, `B08`, `B8A`, `B09`, `B10`, `B11`, `B12`.
- **Mixed-Resolution Co-Registration:** Native Sentinel-2 GeoTIFFs mix `10 m` (`B02`, `B03`, `B04`, `B08`), `20 m` (`B05`–`B07`, `B8A`, `B11`, `B12`), and `60 m` (`B01`, `B09`, `B10`) grids. [`OSCDDataset._load_temporal_image`](file:///Users/ishaan/satellite/ml/datasets/oscd.py#L218) automatically selects a `10 m` band (`B04`, `B03`, `B02`, or `B08`) as the spatial reference grid and passes all requested bands through [`load_multispectral_stack`](file:///Users/ishaan/satellite/ml/preprocessing/image_loader.py) so every stacked band matches pixel grid, CRS, and affine transform.
- Default normalization mode for multispectral inputs is `"sentinel2_sr"` (scaling integer reflectance by `10000.0` with finite-pixel clamping).

---

## 4. Dataset C: Custom Satellite / RGB Imagery

### Directory Layout

For batch preparation or training with [`CustomPairDataset`](file:///Users/ishaan/satellite/ml/datasets/custom_dataset.py#L24):

```text
<custom_source_dir>/
├── before/    # PNG, JPG/JPEG, TIF/TIFF
├── after/     # Matching filenames/stems
└── mask/      # Optional binary labels
```

Single pairs can also be uploaded directly through the FastAPI `/api/upload` route or the Streamlit UI.

### Validation Rules Enforced Before Preprocessing or Inference

Implemented in [`ml/preprocessing/image_loader.py`](file:///Users/ishaan/satellite/ml/preprocessing/image_loader.py) and [`ml/preprocessing/alignment.py`](file:///Users/ishaan/satellite/ml/preprocessing/alignment.py):

1. **Allowed Extensions (`SUPPORTED_IMAGE_EXTENSIONS`):** `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`. Any other format fails immediately with `ImageValidationError`.
2. **Corruption & Integrity Checks:** Standard images are verified with `PIL.Image.verify()` before pixel decoding; TIFFs are opened via `rasterio` and checked for NaN/Inf/all-zero invalid masks.
3. **File Size & Dimension Bounds:** Enforces `max_size_mb=50.0`, `min_dim=16`, and `max_dim=20000`.
4. **Channel Consistency:** Single-channel grayscale images can be expanded to 3 channels when 3-band RGB is expected (`warnings` note attached), whereas >3-band multispectral inputs raise `ImageValidationError` if fed to a strict 3-channel configuration without explicit band selection.
5. **Pair Compatibility (`validate_image_pair` & `align_image_pair`):**
   - Verifies channel counts match between `before` and `after` and that neither image has zero valid pixels (raising `ImageValidationError` otherwise).
   - Flags CRS, affine-transform, or dimension mismatches cleanly; when both GeoTIFFs are georeferenced, reprojects `after` onto `before`'s CRS and affine grid via `rasterio.warp.reproject`. For non-georeferenced RGB pairs, optionally applies OpenCV ECC registration or resizes `after` to match `before`, recording every operation in [`AlignmentReport`](file:///Users/ishaan/satellite/ml/preprocessing/alignment.py#L19).

---

## 5. Dataset Verification & Preparation Scripts

### Check Local Dataset Availability

```bash
.venv/bin/python scripts/download_data.py --data-dir data --dataset all
```

Flags:
- `--data-dir`: Root data directory (default: `<repo>/data`).
- `--dataset`: `all` | `levir` | `oscd`.

### Tile & Validate a Dataset into Patches

```bash
# Prepare and tile a custom dataset directory into data/processed
.venv/bin/python scripts/prepare_dataset.py \
  --dataset custom \
  --source-dir /path/to/custom_source \
  --output-dir data/processed/custom_tiles \
  --patch-size 256 \
  --stride 256

# Verify patch counts for OSCD multispectral splits
.venv/bin/python scripts/prepare_dataset.py \
  --dataset oscd \
  --source-dir data/raw/OSCD \
  --bands B04 B03 B02 B08 \
  --patch-size 256 \
  --stride 256
```

For `custom` datasets, [`prepare_custom_dataset`](file:///Users/ishaan/satellite/ml/datasets/custom_dataset.py#L147) validates and aligns every pair, writes `256x256` PNG tiles (or resized masks), and records exact provenance (`source_dir`, `generated_patches`, and alignment method) in `manifest.json`.
