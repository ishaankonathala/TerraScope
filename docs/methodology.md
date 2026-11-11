# Research Methodology

## 1. Experimental Protocol for Fair Comparison (RQ1 & RQ2)

To avoid biased comparisons between CNN and Transformer families:

1. **Identical Data Splits:** All three models (`siamese_unet`, `vit`, `swin`) train and evaluate on exact matching train/val/test file lists (`LEVIR-CD` official splits or `OSCD` official city splits).
2. **Identical Input Resolution & Normalization:** Standard `256 x 256` patches with identical normalization (`imagenet` for RGB, `sentinel2_sr` for Sentinel-2 reflectance).
3. **Paired Geometric Invariance:** Random flips, 90-degree rotations, and crops apply the exact same spatial transforms to `before`, `after`, `ground_truth_mask`, and `valid_mask`.
4. **Unified Loss & Optimization Baseline:** Combined `0.5 * BCE + 0.5 * Dice` loss addressing extreme change/no-change class imbalance, `AdamW` optimizer, `CosineAnnealingLR`, and fixed random seeds (`seed=42`).
5. **Valid-Mask Discipline:** Masked clouds, shadows, and `nodata` pixels are excluded from both loss computation and pixel-level evaluation metrics rather than treated as "unchanged".
6. **Consistent Thresholding:** All binary evaluation metrics (`Precision`, `Recall`, `F1`, `IoU`, `Dice`, `Accuracy`) are computed at identical confidence thresholds (`0.5` unless tuned consistently on validation).

## 2. Investigating Robustness Factors (RQ3)

- **Cloud & Shadow Interference:** Configurable Sentinel-2 `SCL` pixel masking (`classes 0, 1, 3, 8, 9, 10`) prevents false-positive change detections caused by transient cloud cover.
- **Spatial & CRS Misalignment:** `ml/preprocessing/alignment.py` enforces CRS/transform matching via `rasterio.warp.reproject` and provides optional OpenCV ECC intensity registration to evaluate misalignment sensitivity.
- **Seasonal Shifts & NDVI:** `ml/satellite/indices.py` computes supplementary `(B8 - B4) / (B8 + B4)` vegetation indicators while explicitly documenting that phenological NDVI changes do not alone prove structural land-use conversion.

## 3. Interpretability Evaluation (RQ4)

Architecture-appropriate explanations (`Grad-CAM`, `Attention Rollout`, `Swin Window Attention`) are extracted solely from actual internal tensors/weights during forward/backward passes, never synthesized or approximated from prediction masks.

