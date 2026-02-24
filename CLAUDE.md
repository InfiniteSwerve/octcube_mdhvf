# CLAUDE.md - OCTCube MDHVF Project Context

## Project Summary

Predicting **Humphrey Visual Field Mean Deviation (HVF MTD)** from **3D OCT volumes** using OCTCube, a pretrained 3D vision transformer foundation model for optical coherence tomography.

**Clinical goal**: Predict degree of visual field loss from retinal imaging, enabling earlier and more objective glaucoma assessment.

**Paper reference**: "OCTCube: a 3D foundation model for optical coherence tomography" (arXiv:2408.11227)
**Reference implementation**: https://github.com/ZucksLiu/OCTCubeM

## Repository Layout

```
octcube.py         # Core model: encoder (OCTCubeViT, OCTCubeWrapper), decoder (MAEDecoder),
                   #   task heads (BetaRegressionHead, OCTCubeSegHead),
                   #   analysis tools (PatchAnalyzer, SteeringVectorExtractor)
main_octcube.py    # Training script, metrics, checkpointing, entry point
dataset.py         # HVFDataset: loads DICOM OCTs + TSV labels, patient-level splits
utils.py           # load_config() helper
config.json        # Local config (gitignored) - must contain {"dcm_path": "..."}
```

## How to Run

```bash
# Install deps
uv sync

# Ensure config.json exists with dcm_path pointing to DICOM storage
cp config.example.json config.json  # then edit dcm_path

# Run (currently does feature extraction; toggle by uncommenting training loop)
python main_octcube.py
```

## Architecture

### Encoder: OCTCubeViT (pretrained, ~300M params for large)
- Input: `(B, T=48, C=1, H=512, W=512)` grayscale OCT volumes
- 3D patch embedding via Conv3d: spatial 16x16, temporal 3 frames
- Produces 16,384 tokens (16 temporal x 1024 spatial) of dim 1024 (large) or 768 (base)
- Separate sine-cosine positional embeddings for spatial and temporal dims
- 24 transformer blocks (large) / 12 (base), with stochastic depth

### Regression Head: BetaRegressionHead
- AttentionPool (learned attention over all 16k tokens) -> single 1024-dim vector
- MLP (1024 -> 256 -> 2) with Softplus activation -> (alpha, beta) params
- Models prediction as Beta(alpha, beta) distribution over [0, 1]
- Loss: negative log-likelihood of Beta distribution

### Other Heads (available but not primary focus)
- **OCTCubeSegHead**: ConvNeXt-based per-slice segmentation
- **MAEDecoder**: 8-block decoder for reconstruction (used by PatchAnalyzer)
- **SteeringVectorExtractor**: computes activation directions between pathological/normal patches

## Data Pipeline

### HVFDataset (dataset.py)
- **Source**: TSV files (`macula_oct_partially_deduplicated.tsv` or `optic_nerve_oct_partially_deduplicated.tsv`) + DICOM files
- **Splits**: 70/15/15 train/val/test by patient MRN (persisted to `training_splits.csv`)
- **Label normalization**: HVF MTD rescaled to [0, 1] via percentile-based bounds, clipped to [1e-6, 1-1e-6] for Beta distribution stability
- **Output per sample**: `{"frames": (1, 48, 512, 512), "label": scalar in [0,1], "mrn": str}`

### Known Issues
- **dataset.py:62-63**: `mx`/`mn` variable names are swapped (mx holds min, mn holds max), and margin is applied twice (bug). The normalization still works due to the way rescale_label uses them, but the double margin gives ~10% padding instead of the intended 5%.

## Training Configuration (TrainConfig dataclass)

| Parameter | Default | Notes |
|-----------|---------|-------|
| batch_size | 1 | Memory-constrained (16k tokens per volume) |
| num_frames | 48 | OCT slices per volume |
| img_size | 512 | Spatial resolution |
| patch_size | 16 | -> 32x32 = 1024 spatial patches |
| t_patch_size | 3 | -> 16 temporal patches |
| model_size | 'large' | 1024-dim, 24 blocks, 16 heads |
| epochs | 20 | |
| checkpoint_path | /storage2/.../OCTCube.pth | Pretrained encoder weights |

**Optimizer**: AdamW with separate LRs - encoder 1e-5, head 1e-3
**Mixed precision**: GradScaler with gradient clipping (max_norm=1.0)

## Current Project State

The training loop was used for end-to-end fine-tuning (encoder + regression head), then the script was switched to **feature extraction mode** (lines 483-499 of main_octcube.py). Features from the frozen encoder are saved as `features.npy` (shape: N x 1024) and `labels.npy`.

**Next steps**: Experimenting with both approaches:
1. Training lightweight models on extracted features (linear probe, MLP, etc.)
2. End-to-end fine-tuning with hyperparameter sweeps

The `validation_epoch` / `validation_partial_epoch` functions are stale (still use segmentation-era shapes) and need updating for regression before being re-enabled.

## Key Patterns & Conventions

- **Type checking**: `@jaxtyped(typechecker=beartype)` with `Float[Tensor, "B N D"]` shape annotations
- **Tensor dimension conventions**: Dataset outputs `(C, T, H, W)` per sample -> batched to `(B, C, T, H, W)` -> model rearranges to `(B, T, C, H, W)` for encoder
- **Loss reporting**: `np.exp(loss.item())` is logged (exponentiated NLL), not raw loss
- **Metrics**: rolling window (100) of predictions for MAE and Pearson r
- **Checkpoints**: Only head state_dict is saved (encoder is loaded from pretrained)

## Output Artifacts

| File | Description |
|------|-------------|
| `features.npy` | Encoder features, shape (N, 1024) |
| `labels.npy` | Normalized labels, shape (N,) |
| `checkpoints_octcube/latest.pt` | Training checkpoint (head weights + optimizer + metrics) |
| `checkpoints_octcube/best.pt` | Best checkpoint (not currently tracked) |
| `octcube_metrics.json` | Metrics history |
| `octcube_metrics.png` | Loss/metric curves |
| `regression_scatter.png` | Pred vs GT scatter plot |
| `training_splits.csv` | Patient-level train/val/test assignments |

## Dependencies

Python 3.13, PyTorch 2.10+, einops, jaxtyping, beartype, pandas, pydicom, numpy, matplotlib, tqdm.
Package manager: uv (`uv sync` to install).
