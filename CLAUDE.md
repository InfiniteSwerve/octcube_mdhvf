# CLAUDE.md — OCTCube MDHVF Project Context

## What This Project Does

Predicts **HVF (Humphrey Visual Field) Mean Total Deviation (MTD)** from **3D OCT (Optical Coherence Tomography) volumes** using **OCTCube**, a pretrained 3D Vision Transformer foundation model.

**Clinical goal:** Given a retinal OCT scan (a 3D volume of ~48 grayscale slices at 512x512), predict the patient's visual field metric — a bounded scalar in [0, 1] after normalization.

**Paper reference:** "OCTCube: a 3D foundation model for optical coherence tomography" (arXiv:2408.11227)

## Architecture Overview

```
DICOM OCT Volume (B, 1, 48, 512, 512)
    ↓ permute to (B, 48, 1, 512, 512)
    ↓
PatchEmbed3D: 3D conv (t_patch=3, patch=16)
    → (B, 16, 1024, 1024)  [16 temporal patches, 32×32=1024 spatial patches, dim=1024]
    ↓
+ spatial pos_embed + temporal pos_embed
    ↓ flatten to (B, 16384, 1024)
    ↓
24× Transformer Blocks (Large config: dim=1024, heads=16)
    ↓
AttentionPool → (B, 1024)
    ↓
BetaRegressionHead: MLP → (α, β) parameters
    ↓
Loss: Beta NLL = -Beta(α, β).log_prob(target)
```

**Key design choice:** Uses a **Beta distribution** for regression since targets are bounded in (0, 1). The model outputs α and β parameters, and prediction = Beta(α, β).mean.

## File Map

| File | Lines | Purpose |
|---|---|---|
| `octcube.py` | 1783 | All model architectures: ViT encoder, MAE decoder, segmentation head, Beta regression head, steering vectors, patch analyzer |
| `main_octcube.py` | 552 | Training pipeline, metrics, checkpointing. **Currently exits early at line 499 to extract features instead of training.** |
| `dataset.py` | 104 | `HVFDataset` — loads DICOM OCT files, pairs with HVF MTD labels from TSV, patient-level train/val/test splits (70/15/15) |
| `utils.py` | 5 | `load_config()` — reads `config.json` into SimpleNamespace |
| `config.json` | — | Must contain `{"dcm_path": "path/to/dicom/files"}` (gitignored, see `config.example.json`) |

## Key Classes in octcube.py

- **`OCTCubeViT`** (line 307): Core 3D ViT encoder with separate spatial/temporal positional embeddings
- **`OCTCubeWrapper`** (line 524): Factory for base/large configs, handles pretrained checkpoint loading with flash attention key remapping
- **`OCTCubeMAE`** (line 842): Masked autoencoder (encoder + decoder) for self-supervised pretraining
- **`PatchAnalyzer`** (line 1076): Per-patch reconstruction loss analysis for anomaly detection
- **`SteeringVectorExtractor`** (line 1198): Extracts/applies steering vectors from unusual vs normal patches
- **`OCTCubeSegmenter`** (line 1524): Segmentation pipeline with ConvNeXt upsampling head
- **`OCTCubeRegression`** (line 1662): **Active model** — encoder + AttentionPool + BetaRegressionHead
- **`BetaRegressionHead`** (line 1629): AttentionPool → Linear(1024, 256) → ReLU → Dropout(0.1) → Linear(256, 2) → Softplus → (α, β)

## Model Configs

| Config | embed_dim | depth | heads |
|---|---|---|---|
| base | 768 | 12 | 12 |
| large | 1024 | 24 | 16 |

Current default: **large**

## Current State of `main_octcube.py`

The `full_supervised_run()` function currently:
1. Creates train/val/test DataLoaders
2. Initializes `OCTCubeRegression` with pretrained weights from `/storage2/fs1/leeay/Active/jstrand/projects/OCTCubeM/ckpt/OCTCube.pth`
3. **Extracts features** (mean-pooled encoder tokens) for all training data → saves `features.npy` and `labels.npy`
4. **Exits** (`exit()` at line 499) — the training loop below never runs

The training loop (lines 501-548) is present but bypassed. It uses:
- AdamW with separate LRs: encoder=1e-5, head=1e-3
- Mixed precision (AMP + GradScaler)
- Gradient clipping (max_norm=1.0)
- Beta NLL loss
- Rolling regression metrics (MAE, Pearson r) with matplotlib plots

Validation epochs are also commented out (lines 289-290, 538-539, 544-545).

## Dataset Details

- **Source data:** DICOM files containing multi-frame OCT volumes
- **Labels:** `hvf_mtd` from TSV files (`macula_oct_partially_deduplicated.tsv` or `optic_nerve_oct_partially_deduplicated.tsv`)
- **Label normalization:** 1st-99th percentile scaling with 5% margin, clipped to (1e-6, 1-1e-6)
- **Image normalization:** Per-volume min-max to [0, 1]
- **Split strategy:** Patient-level (by MRN) to avoid data leakage, persisted in `training_splits.csv`
- **Merge key quirk:** Dataset merges on `hvf_mrn` ↔ `mrn` columns (line 57)
- **Anatomy:** Supports "macula" (default) and "optic_nerve"

## Build / Run

```bash
# Dependencies (uses uv)
uv sync

# Run training / feature extraction
python main_octcube.py

# Test model shapes
python octcube.py
```

Requires: `config.json` with DICOM path, the TSV label files, and optionally the pretrained OCTCube checkpoint.

## Type Checking

Uses `jaxtyping` + `beartype` for runtime tensor shape checking throughout. The decorator pattern:
```python
typechecked = jaxtyped(typechecker=beartype)
```

## Gotchas / Things to Watch

1. **Dimension ordering:** Dataset returns `(1, F, H, W)` but the model expects `(B, C, T, H, W)` then internally permutes `(B, T, C, H, W) → (B, C, T, H, W)` — the permute at `octcube.py:1721` swaps dims 1 and 2, which works because C=1 and the input from the dataset has channel first
2. **`beta_nll_loss` is defined twice** — once in `octcube.py` (line 1652) and once in `main_octcube.py` (line 252). The import at `main_octcube.py:28` imports from octcube but the local definition shadows it
3. **`batch_size` and `num_workers` in TrainConfig** are class variables (no type annotation), not dataclass fields — they work but won't appear in `__init__`
4. **Metrics class references undefined attributes:** `TrainConfig.dice_calc_interval` and `TrainConfig.volume_report_interval` (lines 188, 191) — leftover from segmentation code, will error if called
5. **`rescale_label` naming:** `self.mx` is assigned `a - margin` (the min) and `self.mn` is assigned `b + margin` (the max) — naming is inverted but the math in `rescale_label` is consistent with itself
6. **No random seed setting** in dataset split creation or anywhere in training
7. **Pretrained checkpoint path** is hardcoded to a specific storage path
