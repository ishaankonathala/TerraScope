# Project Defense & Viva Preparation Notes

Concise technical rationale, design trade-offs, and implementation defenses aligned directly with the codebase.

---

## 1. Core Research Question

**Q: Do Vision Transformers automatically outperform CNNs on satellite change detection?**  
**A:** No — and this project avoids assuming that. CNNs (`SiameseUNet`) retain strong local translation equivariance and crisp inductive bias on smaller datasets (such as OSCD's 14 training cities). Global self-attention in `ViTChangeDetector` captures long-range context across whole scenes but operates at coarser `16x16` patch tokens, while `SwinChangeDetector` bridges both via hierarchical `H/4 -> H/32` shifted-window representations. Only controlled benchmarking under identical splits, losses, and seeds (`benchmark_models`) establishes which wins on a given spatial resolution and dataset size.

---

## 2. Architectural Design Decisions

1. **True Siamese Weight Sharing ([`BaseChangeDetector`](file:///Users/ishaan/satellite/ml/models/blocks.py)):**  
   Before ($T_1$) and after ($T_2$) images pass through the **exact same shared backbone weights**. Feeding stacked 6-channel `[before; after]` directly into layer 1 mixes raw seasonal/illumination differences prematurely, whereas Siamese encoders align semantic features first and apply [`TemporalFusionBlock`](file:///Users/ishaan/satellite/ml/models/blocks.py) (`|F2 - F1|` + concatenation projection) across stages.
2. **Non-Overlapping CNN vs. ViT vs. Swin Feature Extraction:**
   - [`SiameseUNet`](file:///Users/ishaan/satellite/ml/models/siamese_unet.py): `timm` ResNet encoder (`features_only=True`) or offline 4-stage `_LightweightCNNEncoder` fallback + U-Net decoder.
   - [`ViTChangeDetector`](file:///Users/ishaan/satellite/ml/models/vit_change_detector.py#L22): Extracts multi-depth intermediate token grids from `vit_tiny_patch16_224` (`dynamic_img_size=True`), reshapes tokens back into 2D grids at multiple Transformer depths, fuses temporal differences, and progressively upsamples with spatial refinement (`ConvBNAct` / `DecoderUpBlock`) to suppress checkerboard patch artifacts.
   - [`SwinChangeDetector`](file:///Users/ishaan/satellite/ml/models/swin_change_detector.py#L21): Captures native 4-stage hierarchical feature maps from `swin_tiny_patch4_window7_224` (`feature_info` metadata), giving fine `H/4` edge detail plus shifted-window cross-patch interaction.
3. **Arbitrary Input Dimensions & Multispectral Compatibility:**  
   Models pad odd/non-multiple-of-32 image dimensions symmetrically (`F.pad`) before forward passes and crop logits cleanly back to the input spatial extent. Band counts >3 (e.g., 4-band RGB+NIR or 13-band Sentinel-2) adapt `in_chans=in_channels` in `timm` cleanly.

---

## 3. Explainability (XAI) Rigor & Honest Limitations

**Q: How do you ensure XAI visualizations are real and not post-hoc edge filters?**  
**A:** Each explainer inspects actual live tensors inside PyTorch:
- **CNN (`compute_siamese_gradcam` in [`gradcam.py`](file:///Users/ishaan/satellite/ml/explainability/gradcam.py#L15)):** Hooks activations and exact backward gradients ($\partial \hat{y}_{\text{change}} / \partial A^k$) at the deepest shared convolutional decoder/fusion feature layer.
- **ViT (`attention_rollout.py`):** Temporarily disables `fused_attn` on `timm` `Attention` modules, captures post-softmax `attn_drop` attention matrices from every Transformer block for $T_1$ and $T_2$, strips prefix tokens (`[CLS]` / register tokens), and applies Abnar & Zuidema (2020) residual matrix multiplication $(0.5 A^{(l)} + 0.5 I)$ recursively across layers.
- **Swin (`attention_rollout.py`):** Captures shifted `WindowAttention` post-softmax matrices from `swin_tiny` blocks, aggregates query-attended token importances within local windows, and folds window scores back onto the spatial feature map before bilinear upsampling.

All UI views and outputs display `INTERPRETABILITY_DISCLAIMER`, clarifying that saliency and rollout maps indicate model attention patterns rather than causal land-use proof.

---

## 4. Geospatial & Remote Sensing Integrity

1. **No Naive Pixel Counting in Geographic Coordinates:**  
   [`compute_pixel_areas_m2`](file:///Users/ishaan/satellite/ml/geospatial/analysis.py#L22) inspects `pyproj.CRS`. Projected CRS rasters use the affine determinant $\lvert a \cdot e - b \cdot d \rvert$ times axis linear-unit squared; geographic lat/lon CRS rasters (`EPSG:4326`) calculate latitude-dependent geodesic cell areas along every row using `pyproj.Geod(ellps="WGS84")` rather than treating degrees as meters. Non-georeferenced PNG/JPG uploads explicitly return `"image_relative_only"` without fabricated square kilometers.
2. **Temporal Grid & Multi-Resolution Co-Registration:**  
   Matching `(H, W)` pixel arrays never imply identical ground alignment. When both inputs are georeferenced GeoTIFFs, [`align_image_pair`](file:///Users/ishaan/satellite/ml/preprocessing/alignment.py#L192) reprojects `after` directly onto `before`'s CRS and affine transform (`rasterio.warp.reproject`). In OSCD and GEE Sentinel-2 workflows, `20 m` SWIR/SCL and `60 m` bands are explicitly aligned onto the target `10 m` reference grid (`reproject(crs=target_crs, scale=clean_scale)`).
3. **SCL Cloud Masking & NDVI Non-Causality:**  
   Sentinel-2 `SCL` masks classes `(1, 3, 8, 9, 10)` at `20 m`, so sub-pixel clouds and shadows can still affect `10 m` bands (`NDVI_SCIENTIFIC_NOTE`). Likewise, `NDVI = (B8 - B4) / (B8 + B4)` highlights spectral vigor deltas that may stem from phenology or harvest rather than permanent land-use change.
4. **Honest Benchmark Reporting:**  
   Neither public datasets nor test metrics are fabricated (`is_synthetic` flags track smoke tests separately from real evaluations, and untrained backbones clearly display `"untrained"` in the UI and API responses).

