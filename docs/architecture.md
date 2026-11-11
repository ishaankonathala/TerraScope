# System Architecture Documentation

## Modular Design Overview

The codebase enforces strict separation of concerns across five subsystems:

```text
satellite/
├── ml/                  # Pure deep-learning, preprocessing, geospatial, XAI & GEE logic
│   ├── models/          # SiameseUNet, ViTChangeDetector, SwinChangeDetector, model_factory
│   ├── datasets/        # LEVIR-CD, OSCD, CustomPairDataset, PairedChangeTransform
│   ├── preprocessing/   # Image validation, reprojection alignment, SCL masking, normalization, tiling
│   ├── satellite/       # Optional Google Earth Engine Sentinel-2 client, composites, NDVI, export
│   ├── training/        # Losses, optimizers, schedulers, callbacks, resume-capable trainer
│   ├── evaluation/      # Pixel metrics, fair 3-model benchmark runner, comparison plots
│   ├── explainability/  # ViT Attention Rollout, Swin Window Attention, CNN Grad-CAM
│   ├── inference/       # Overlapping Hann-blended tiled ChangePredictor & post-processing
│   ├── geospatial/      # CRS-aware / geodesic area calculation, GeoTIFF export, Folium maps
│   └── utils/           # Device detection (CPU/MPS/CUDA), seeding, peak memory tracking
├── backend/             # FastAPI REST service, Pydantic schemas, LRU model cache, PDF/CSV reports
├── frontend/            # 8-page Streamlit Geospatial AI dashboard with Dark/Light theme
├── scripts/             # CLI utilities for dataset check, preparation, training, evaluation, demo
├── experiments/         # Configs, real training/benchmark JSON/CSV results, TensorBoard logs
├── outputs/             # Change masks, XAI heatmaps, PDF reports, GEE export manifests
└── docs/                # Academic literature, methodology, dataset & GEE guides, viva notes
```

## Model Architectures

### 1. Siamese U-Net (`ml/models/siamese_unet.py`)
- **Encoder:** Shared CNN backbone (`timm` `resnet18` by default, or lightweight `custom_cnn`).
- **Temporal Comparison:** Multi-scale `TemporalFusionBlock` concatenating `[|f_t1 - f_t2|, f_t1, f_t2]` projected through `1x1` and `3x3` Conv-BN-GELU layers.
- **Decoder:** U-Net bilinear upsampling with skip connections from all encoder scales to `(B, num_classes, H, W)` logits.
- **Explainability:** Grad-CAM attached to the final spatial decoder refinement block.

### 2. Vision Transformer Change Detector (`ml/models/vit_change_detector.py`)
- **Encoder:** Shared `timm` Vision Transformer (`vit_tiny_patch16_224` with `dynamic_img_size=True`).
- **Spatial Token Reassembly:** Extracts intermediate spatial patch-token feature maps `[B, D, H/16, W/16]` from 4 distributed Transformer blocks (`[2, 5, 8, 11]` for 12-layer ViT), fuses temporal features at each depth, and upsamples into a multi-resolution feature pyramid (`H/2, H/4, H/8, H/16`).
- **Explainability:** Disables `fused_attn` temporarily during inspection to capture exact post-softmax self-attention matrices `[B, heads, N, N]` for Abnar & Zuidema Attention Rollout.

### 3. Swin Transformer Change Detector (`ml/models/swin_change_detector.py`)
- **Encoder:** Shared `timm` Swin Transformer (`swin_tiny_patch4_window7_224` with `dynamic_img_size=True`) yielding 4 true hierarchical stages `(H/4, H/8, H/16, H/32)`.
- **Decoder:** Fuses temporal features across all 4 hierarchical scales via `TemporalFusionBlock` and reconstructs full-resolution logits through an FPN/U-Net decoder.
- **Explainability:** Captures `WindowAttention` softmax weights across all layers and un-partitions shifted windows back onto spatial stage grids.

