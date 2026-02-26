"""Evaluate how sensitive the BetaRegressionHead is to Gaussian noise on encoder features.

Loads phase1_head.pt and the pre-extracted features, adds N(0, sigma^2) noise
for a range of sigma values, and plots how alpha and beta change.

Produces two plots:
  1. Mean alpha and beta vs sigma
  2. Relative change in alpha and beta vs sigma (% change from sigma=0 baseline)

Usage:
    python noise_sensitivity.py
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from octcube import BetaRegressionHead

# ---------------------------------------------------------------------------
# Config — matches main_octcube.py TrainConfig defaults
# ---------------------------------------------------------------------------
FEATURE_DIR = "extracted_features"
FEATURE_POOL = "mean_raw"
SAVE_DIR = "checkpoints_octcube"
EMBED_DIM = 1024  # 'large' model
SIGMAS = np.concatenate([
    np.linspace(0, 0.5, 11),   # fine resolution near 0
    np.linspace(0.75, 3.0, 10),
])
SIGMAS = np.unique(np.round(SIGMAS, 4))


def main():
    # Load head
    head_path = f"{SAVE_DIR}/phase1_head.pt"
    head = BetaRegressionHead(in_features=EMBED_DIM, hidden_dim=256).cuda()
    head.load_state_dict(torch.load(head_path, map_location="cuda", weights_only=True))
    head.eval()
    print(f"Loaded head from {head_path}")

    # Load features (use val set — unseen during phase 1 training)
    feats = torch.from_numpy(
        np.load(f"{FEATURE_DIR}/{FEATURE_POOL}_val.npy")
    ).cuda()
    labels = np.load(f"{FEATURE_DIR}/labels_val.npy")
    N = len(feats)
    print(f"Loaded {N} val features, shape {feats.shape}")

    # Feature norm stats (useful context for interpreting sigma)
    feat_norms = feats.norm(dim=-1)
    print(f"Feature L2 norms: mean={feat_norms.mean():.2f}, "
          f"std={feat_norms.std():.2f}, "
          f"min={feat_norms.min():.2f}, max={feat_norms.max():.2f}")

    # Run noisy evaluation
    mean_alphas = []
    mean_betas = []
    std_alphas = []
    std_betas = []

    with torch.no_grad():
        for sigma in SIGMAS:
            noise = torch.randn_like(feats) * sigma
            noisy_feats = feats + noise

            alpha, beta = head(noisy_feats)  # (N,), (N,)
            mean_alphas.append(alpha.mean().item())
            mean_betas.append(beta.mean().item())
            std_alphas.append(alpha.std().item())
            std_betas.append(beta.std().item())

            if sigma == 0 or sigma == SIGMAS[-1] or abs(sigma - 1.0) < 0.01:
                pred_mean = (alpha / (alpha + beta)).mean().item()
                print(f"  sigma={sigma:.3f}: alpha={alpha.mean():.4f}±{alpha.std():.4f}, "
                      f"beta={beta.mean():.4f}±{beta.std():.4f}, "
                      f"pred_mean={pred_mean:.4f}")

    mean_alphas = np.array(mean_alphas)
    mean_betas = np.array(mean_betas)
    std_alphas = np.array(std_alphas)
    std_betas = np.array(std_betas)

    # Baseline (sigma=0)
    a0, b0 = mean_alphas[0], mean_betas[0]

    # -----------------------------------------------------------------------
    # Plot 1: Absolute alpha and beta vs sigma
    # -----------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(SIGMAS, mean_alphas, "o-", color="tab:blue", label="mean alpha", markersize=4)
    ax1.fill_between(SIGMAS, mean_alphas - std_alphas, mean_alphas + std_alphas,
                     alpha=0.2, color="tab:blue")
    ax1.plot(SIGMAS, mean_betas, "s-", color="tab:orange", label="mean beta", markersize=4)
    ax1.fill_between(SIGMAS, mean_betas - std_betas, mean_betas + std_betas,
                     alpha=0.2, color="tab:orange")
    ax1.set_xlabel("Noise σ")
    ax1.set_ylabel("Parameter value")
    ax1.set_title("BetaRegressionHead: α and β vs Gaussian noise σ")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(a0, color="tab:blue", ls="--", alpha=0.4)
    ax1.axhline(b0, color="tab:orange", ls="--", alpha=0.4)

    # -----------------------------------------------------------------------
    # Plot 2: Relative change (%) from baseline
    # -----------------------------------------------------------------------
    rel_alpha = (mean_alphas - a0) / (a0 + 1e-8) * 100
    rel_beta = (mean_betas - b0) / (b0 + 1e-8) * 100

    ax2.plot(SIGMAS, rel_alpha, "o-", color="tab:blue", label="Δα / α₀  (%)", markersize=4)
    ax2.plot(SIGMAS, rel_beta, "s-", color="tab:orange", label="Δβ / β₀  (%)", markersize=4)
    ax2.axhline(0, color="gray", ls="--", alpha=0.5)
    ax2.set_xlabel("Noise σ")
    ax2.set_ylabel("Relative change (%)")
    ax2.set_title("Relative change in α and β from clean features")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = "noise_sensitivity.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    main()
