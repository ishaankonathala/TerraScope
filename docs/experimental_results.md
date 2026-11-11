# Experimental Results & Benchmarking Protocol

## 1. Benchmark Status Summary

> [!IMPORTANT]
> Public benchmark datasets (**LEVIR-CD** and **OSCD**) are external datasets subject to their respective academic licenses and were **not** downloaded or trained on in this workspace snapshot. Accordingly, all real-dataset benchmark tables below explicitly report **`Not evaluated`** rather than fabricating or extrapolating metrics.

---

## 2. Controlled Evaluation Protocol

To answer our primary research objective rigorously — *whether Transformer-based architectures provide measurable advantages over CNN baselines under identical test data and evaluation procedures* — all three models share:

1. **Identical Splits & Patch Grid:** Same train/val/test partitions, `256x256` tile size, and non-overlapping test evaluation grid via [`benchmark_models`](file:///Users/ishaan/satellite/ml/evaluation/benchmark.py).
2. **Unified Optimization Config ([`experiments/configs/default_benchmark.yaml`](file:///Users/ishaan/satellite/experiments/configs/default_benchmark.yaml)):**
   - Seed: `42`
   - Optimizer: `AdamW` (`lr=3e-4`, `weight_decay=1e-4`), Cosine annealing scheduler
   - Loss: `bce_dice`
   - Epochs: `20` (with early stopping patience `8`)
   - Decision threshold: `0.5`
3. **Standardized Pixel-Level Metrics ([`ChangeDetectionMetrics`](file:///Users/ishaan/satellite/ml/evaluation/metrics.py#L13)):**
   - `iou` (Changed-Class Intersection-over-Union / Jaccard Index)
   - `f1_score` & `dice`
   - `precision`, `recall`, `specificity`
   - `accuracy` & `macro_f1`
   - `tp`, `fp`, `fn`, `tn`, `total_valid_pixels`
4. **System Efficiency Metrics:** Parameter count, model checkpoint size (`MB`), per-batch / per-sample inference latency (`s`), and peak memory (`MB`).

---

## 3. Experimental Tables (Pending Real Dataset Acquisition)

### Table 1: LEVIR-CD Test Split (`1024x1024` tiled to `256x256`, RGB)

| Model | Backbone | Status | IoU ↑ | F1 ↑ | Precision ↑ | Recall ↑ | Accuracy ↑ | Latency (s/img) ↓ | Params (M) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Siamese U-Net** | `resnet18` | `Not evaluated` | — | — | — | — | — | — | — |
| **ViT Change Detector** | `vit_tiny_patch16_224` | `Not evaluated` | — | — | — | — | — | — | — |
| **Swin Change Detector** | `swin_tiny_patch4_window7_224` | `Not evaluated` | — | — | — | — | — | — | — |

### Table 2: OSCD Test Split (10 Cities, `256x256` patches, 3-Band RGB `B04, B03, B02`)

| Model | Backbone | Status | IoU ↑ | F1 ↑ | Precision ↑ | Recall ↑ | Accuracy ↑ | Latency (s/img) ↓ | Params (M) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Siamese U-Net** | `resnet18` | `Not evaluated` | — | — | — | — | — | — | — |
| **ViT Change Detector** | `vit_tiny_patch16_224` | `Not evaluated` | — | — | — | — | — | — | — |
| **Swin Change Detector** | `swin_tiny_patch4_window7_224` | `Not evaluated` | — | — | — | — | — | — | — |

### Table 3: OSCD Multispectral Test Split (`B02, B03, B04, B08` or 13 Bands — Optional Extension)

| Model | Input Bands | Status | IoU ↑ | F1 ↑ | Precision ↑ | Recall ↑ | Notes |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Siamese U-Net** | 4-band / 13-band | `Not evaluated` | — | — | — | — | Uses adapted `in_channels` stem |
| **ViT Change Detector** | 4-band / 13-band | `Not evaluated` | — | — | — | — | Uses adapted patch-embed `in_chans` |
| **Swin Change Detector** | 4-band / 13-band | `Not evaluated` | — | — | — | — | Uses adapted patch-embed `in_chans` |

---

## 4. Research Questions Evaluation Framework

Once checkpoints are trained on LEVIR-CD and OSCD, the comparison dashboard (`03_Model_Comparison.py`) and benchmark JSON/CSV logs evaluate four pre-registered research questions:

- **RQ1 (Quantitative Accuracy):** Under identical hyperparameters and splits, which architecture maximizes changed-class `iou` and `f1_score`?
- **RQ2 (Spatial Boundary vs. Global Context):** Do Swin shifted windows recover sharper building edges than ViT's `16x16` patch tokens, and does CNN inductive bias remain competitive on small datasets like OSCD?
- **RQ3 (Hardware Efficiency):** How do inference latency and peak RAM compare on local `Apple Silicon M1 (8 GB)` vs. discrete CUDA GPUs?
- **RQ4 (Explainability Faithfulness):** Do ViT attention rollout maps and Swin window attention highlight changed structures more coherently than shallow CNN Grad-CAM heatmaps, and where do both still fail?

---

## 5. Commands to Populate Real Benchmarks

After placing datasets in `data/raw/LEVIR-CD` or `data/raw/OSCD` (verified via `scripts/download_data.py`):

```bash
# 1. Train each architecture on LEVIR-CD with identical settings
for model in siamese_unet vit swin; do
  .venv/bin/python scripts/train_model.py \
    --model "$model" \
    --dataset LEVIR-CD \
    --data-dir data/raw/LEVIR-CD \
    --epochs 20 \
    --batch-size 4 \
    --patch-size 256
done

# 2. Run controlled 3-model evaluation on the identical test split
.venv/bin/python scripts/evaluate_model.py \
  --dataset LEVIR-CD \
  --data-dir data/raw/LEVIR-CD \
  --split test \
  --patch-size 256 \
  --batch-size 4
```

Results are automatically written to `experiments/results/` (JSON + CSV) and comparison charts to `outputs/visualizations/`, which populate the Streamlit **Dashboard** and **Model Comparison** pages without manual table edits.

