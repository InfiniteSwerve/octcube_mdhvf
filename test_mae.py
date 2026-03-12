"""
Test MAE reconstruction: load pretrained OCTCubeMAE, reconstruct B-scans,
and save side-by-side original vs reconstructed slices + per-patch error maps.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from dataset import HVFDataset
from octcube import OCTCubeMAE, PatchAnalyzer
from dataclasses import dataclass

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = "mae_reconstructions"


@dataclass
class MAETestConfig:
    checkpoint_path: str = "/storage2/fs1/leeay/Active/jstrand/projects/OCTCubeM/ckpt/OCTCube.pth"
    img_size: int = 256
    patch_size: int = 16
    num_frames: int = 48
    t_patch_size: int = 3
    model_size: str = "large"
    center_crop_frac: float = 0.5
    num_volumes: int = 5        # How many volumes to reconstruct
    slices_per_volume: int = 8  # Which B-scan indices to visualize per volume


def build_mae(cfg: MAETestConfig) -> OCTCubeMAE:
    model = OCTCubeMAE(
        img_size=cfg.img_size,
        patch_size=cfg.patch_size,
        in_chans=1,
        num_frames=cfg.num_frames,
        t_patch_size=cfg.t_patch_size,
        size=cfg.model_size,
        checkpoint_path=cfg.checkpoint_path,
    )
    model.eval().to(DEVICE)
    return model


def save_bscan_grid(original, reconstructed, patch_losses, volume_idx, cfg, out_dir):
    """
    Save a figure showing original / reconstructed / error for selected B-scans.

    original:      (T, C, H, W) tensor
    reconstructed: (T, C, H, W) tensor
    patch_losses:  (T_patches, L_patches) tensor
    """
    T = original.shape[0]
    # Pick evenly-spaced slices
    n = min(cfg.slices_per_volume, T)
    indices = np.linspace(0, T - 1, n, dtype=int)

    fig, axes = plt.subplots(n, 3, figsize=(12, 3 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    p = cfg.patch_size
    t_p = cfg.t_patch_size
    T_patches = T // t_p
    grid = cfg.img_size // p  # spatial grid per dim

    for row, t_idx in enumerate(indices):
        orig_slice = original[t_idx, 0].cpu().numpy()
        recon_slice = reconstructed[t_idx, 0].cpu().numpy()
        error = np.abs(orig_slice - recon_slice)

        axes[row, 0].imshow(orig_slice, cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_ylabel(f"slice {t_idx}", fontsize=9)
        axes[row, 1].imshow(recon_slice, cmap="gray", vmin=0, vmax=1)
        axes[row, 2].imshow(error, cmap="hot", vmin=0, vmax=error.max() * 0.8 + 1e-8)

        # Overlay patch grid on error map
        for ax in axes[row]:
            ax.set_xticks(np.arange(0, cfg.img_size, p), minor=True)
            ax.set_yticks(np.arange(0, cfg.img_size, p), minor=True)
            ax.grid(which="minor", color="white", linewidth=0.3, alpha=0.3)
            ax.tick_params(which="both", bottom=False, left=False,
                           labelbottom=False, labelleft=False)

    axes[0, 0].set_title("Original", fontsize=11)
    axes[0, 1].set_title("Reconstructed", fontsize=11)
    axes[0, 2].set_title("|Error|", fontsize=11)

    fig.suptitle(f"Volume {volume_idx} — MAE reconstruction", fontsize=13)
    fig.tight_layout()
    path = os.path.join(out_dir, f"vol{volume_idx:03d}_bscans.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def save_patch_loss_map(patch_losses, volume_idx, cfg, out_dir):
    """
    Save a heatmap of per-patch reconstruction loss (T_patches x L_patches reshaped to grid).
    """
    T_patches, L_patches = patch_losses.shape
    grid = int(L_patches ** 0.5)
    # Reshape to (T_patches, grid, grid)
    loss_map = patch_losses.cpu().numpy().reshape(T_patches, grid, grid)

    fig, ax = plt.subplots(1, 1, figsize=(10, max(3, T_patches * 0.4)))
    # Tile: each row is a temporal patch, unroll spatial grid horizontally
    # Shape: (T_patches, grid, grid) -> (T_patches * grid, grid) for a tall strip
    tiled = loss_map.reshape(T_patches * grid, grid)
    im = ax.imshow(tiled, cmap="hot", aspect="auto")
    ax.set_xlabel("Spatial patch (x)")
    ax.set_ylabel("Temporal patch × spatial row")
    ax.set_title(f"Volume {volume_idx} — per-patch MSE")
    plt.colorbar(im, ax=ax, fraction=0.02)
    fig.tight_layout()
    path = os.path.join(out_dir, f"vol{volume_idx:03d}_patch_loss.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def main():
    cfg = MAETestConfig()
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading dataset...")
    ds = HVFDataset(
        split_label="val",
        target_size=(cfg.img_size, cfg.img_size),
        normalize=True,
        center_crop_frac=cfg.center_crop_frac,
    )
    print(f"  {len(ds)} validation samples")

    print("Building MAE model...")
    mae = build_mae(cfg)
    analyzer = PatchAnalyzer(mae)

    n = min(cfg.num_volumes, len(ds))
    for i in range(n):
        sample = ds[i]
        frames = sample["frames"]  # (1, T, H, W)
        label = sample["label"]
        mrn = sample["mrn"]
        print(f"\nVolume {i}: MRN={mrn}, HVF_MTD_norm={label:.4f}")

        # (1, T, H, W) -> (1, T, 1, H, W) for the model
        x = frames.unsqueeze(2).to(DEVICE)  # (1, T, C, H, W)

        with torch.no_grad(), torch.amp.autocast("cuda"):
            results = analyzer.compute_patch_losses(x)

        recon = results["reconstructed"]       # (1, T, C, H, W)
        patch_losses = results["patch_losses"]  # (1, T_patches, L_patches)

        # Per-pixel MSE
        mse = ((x - recon) ** 2).mean().item()
        stats = analyzer.get_patch_statistics(patch_losses)
        print(f"  Pixel MSE: {mse:.6f}  |  Patch MSE mean={stats['mean']:.6f}, "
              f"median={stats['median']:.6f}, p95={stats['percentile_95']:.6f}")

        save_bscan_grid(x[0], recon[0], patch_losses[0], i, cfg, OUT_DIR)
        save_patch_loss_map(patch_losses[0], i, cfg, OUT_DIR)

    print(f"\nDone — outputs in {OUT_DIR}/")


if __name__ == "__main__":
    main()
