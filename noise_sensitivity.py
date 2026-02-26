"""Evaluate how sensitive the SimpleRegressionHead is to Gaussian noise on encoder features.

Loads phase1_head.pt and the pre-extracted features, adds noise with ||noise||_2 = sigma
for a range of sigma values, and plots how predictions change.

Produces two plots:
  1. Mean prediction vs sigma
  2. Relative change in prediction vs sigma (% change from sigma=0 baseline)

Usage:
    python noise_sensitivity.py
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from octcube import SimpleRegressionHead

# ---------------------------------------------------------------------------
# Config — matches main_octcube.py TrainConfig defaults
# ---------------------------------------------------------------------------
FEATURE_DIR = "extracted_features"
FEATURE_POOL = "mean_raw"
SAVE_DIR = "checkpoints_octcube"
EMBED_DIM = 1024  # 'large' model
SIGMAS = np.concatenate([
    np.linspace(0, 5, 11),     # fine resolution near 0
    np.linspace(7.5, 50, 10),  # sigma is now ||noise||_2, comparable to feature norms
])
SIGMAS = np.unique(np.round(SIGMAS, 2))


def main():
    # Load head
    head_path = f"{SAVE_DIR}/phase1_head.pt"
    head = SimpleRegressionHead(in_features=EMBED_DIM, hidden_dim=256).cuda()
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
    mean_preds = []
    std_preds = []

    with torch.no_grad():
        for sigma in SIGMAS:
            noise = torch.randn_like(feats)
            noise = noise / noise.norm(dim=-1, keepdim=True) * sigma
            noisy_feats = feats + noise

            pred = head(noisy_feats)  # (N,)
            mean_preds.append(pred.mean().item())
            std_preds.append(pred.std().item())

            if sigma == 0 or sigma == SIGMAS[-1] or abs(sigma - 1.0) < 0.01:
                print(f"  sigma={sigma:.3f}: pred={pred.mean():.4f}±{pred.std():.4f}")

    mean_preds = np.array(mean_preds)
    std_preds = np.array(std_preds)

    # Baseline (sigma=0)
    p0 = mean_preds[0]

    # -----------------------------------------------------------------------
    # Plot 1: Absolute prediction vs sigma
    # -----------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(SIGMAS, mean_preds, "o-", color="tab:blue", label="mean pred", markersize=4)
    ax1.fill_between(SIGMAS, mean_preds - std_preds, mean_preds + std_preds,
                     alpha=0.2, color="tab:blue")
    ax1.set_xlabel("Noise σ (L2 norm)")
    ax1.set_ylabel("Prediction")
    ax1.set_title("SimpleRegressionHead: prediction vs Gaussian noise σ")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(p0, color="tab:blue", ls="--", alpha=0.4)

    # -----------------------------------------------------------------------
    # Plot 2: Relative change (%) from baseline
    # -----------------------------------------------------------------------
    rel_pred = (mean_preds - p0) / (p0 + 1e-8) * 100

    ax2.plot(SIGMAS, rel_pred, "o-", color="tab:blue", label="Δpred / pred₀  (%)", markersize=4)
    ax2.axhline(0, color="gray", ls="--", alpha=0.5)
    ax2.set_xlabel("Noise σ (L2 norm)")
    ax2.set_ylabel("Relative change (%)")
    ax2.set_title("Relative change in prediction from clean features")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = "noise_sensitivity.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    main()
