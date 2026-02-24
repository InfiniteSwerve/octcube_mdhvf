"""
Diagnostic script for regression-to-mean problem.

Run this to:
1. Check if existing features have discriminative signal
2. Extract better features (with fc_norm, multiple pooling strategies)
3. Compare downstream model performance across feature variants
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats


# ── Step 1: Diagnose existing features ──────────────────────────────

def diagnose_existing_features():
    """Check if features.npy actually contains signal for the task."""
    feat_path = Path("features.npy")
    label_path = Path("labels.npy")

    if not feat_path.exists() or not label_path.exists():
        print("features.npy or labels.npy not found — skipping existing feature diagnosis")
        return None, None

    X = np.load(feat_path)
    y = np.load(label_path)
    print(f"Features shape: {X.shape}, Labels shape: {y.shape}")
    print(f"Label range: [{y.min():.4f}, {y.max():.4f}], mean={y.mean():.4f}, std={y.std():.4f}")

    # 1. Check feature variance — are features actually varying?
    feat_std = X.std(axis=0)
    print(f"\nFeature std stats: min={feat_std.min():.6f}, max={feat_std.max():.6f}, "
          f"mean={feat_std.mean():.6f}, median={np.median(feat_std):.6f}")
    dead_features = (feat_std < 1e-6).sum()
    print(f"Dead features (std < 1e-6): {dead_features} / {X.shape[1]}")

    # 2. Per-feature correlation with target
    correlations = np.array([
        stats.pearsonr(X[:, i], y)[0] if feat_std[i] > 1e-6 else 0.0
        for i in range(X.shape[1])
    ])
    correlations = np.nan_to_num(correlations)
    abs_corr = np.abs(correlations)

    print(f"\nPer-feature |correlation| with target:")
    print(f"  max={abs_corr.max():.4f}, mean={abs_corr.mean():.4f}, "
          f"median={np.median(abs_corr):.4f}")
    print(f"  Features with |r| > 0.1: {(abs_corr > 0.1).sum()}")
    print(f"  Features with |r| > 0.2: {(abs_corr > 0.2).sum()}")
    print(f"  Features with |r| > 0.3: {(abs_corr > 0.3).sum()}")

    top_k = 20
    top_idx = np.argsort(abs_corr)[-top_k:][::-1]
    print(f"\nTop {top_k} features by |correlation|:")
    for i, idx in enumerate(top_idx):
        print(f"  [{idx:4d}] r={correlations[idx]:+.4f}")

    # 3. Feature magnitude distribution (checks if fc_norm absence causes issues)
    feat_norms = np.linalg.norm(X, axis=1)
    print(f"\nPer-sample L2 norm: mean={feat_norms.mean():.2f}, std={feat_norms.std():.2f}, "
          f"cv={feat_norms.std()/feat_norms.mean():.4f}")

    feat_means = X.mean(axis=0)
    print(f"Per-dimension mean: range=[{feat_means.min():.4f}, {feat_means.max():.4f}], "
          f"abs_mean={np.abs(feat_means).mean():.4f}")

    # 4. Simple linear probe as sanity check
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    pipe = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    scores = cross_val_score(pipe, X, y, cv=5, scoring='neg_mean_absolute_error')
    print(f"\n5-fold Ridge MAE: {-scores.mean():.4f} +/- {scores.std():.4f}")

    # Compare to dummy (predict mean)
    dummy_mae = np.abs(y - y.mean()).mean()
    print(f"Dummy (predict mean) MAE: {dummy_mae:.4f}")
    print(f"Relative improvement: {(1 - (-scores.mean()) / dummy_mae) * 100:.1f}%")

    # 5. Plot diagnostics
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Label distribution
    axes[0, 0].hist(y, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Normalized label')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Label distribution')
    axes[0, 0].axvline(y.mean(), color='r', linestyle='--', label=f'mean={y.mean():.3f}')
    axes[0, 0].legend()

    # Correlation histogram
    axes[0, 1].hist(abs_corr, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('|Pearson r| with target')
    axes[0, 1].set_ylabel('Feature count')
    axes[0, 1].set_title('Feature-target correlation distribution')

    # Top feature scatter
    best_feat_idx = top_idx[0]
    axes[1, 0].scatter(X[:, best_feat_idx], y, alpha=0.5, s=10)
    axes[1, 0].set_xlabel(f'Feature {best_feat_idx}')
    axes[1, 0].set_ylabel('Label')
    axes[1, 0].set_title(f'Best single feature (r={correlations[best_feat_idx]:+.4f})')

    # PCA projection
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    scatter = axes[1, 1].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='RdYlGn', alpha=0.7, s=10)
    plt.colorbar(scatter, ax=axes[1, 1], label='Label')
    axes[1, 1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    axes[1, 1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    axes[1, 1].set_title('PCA of features (colored by label)')

    plt.tight_layout()
    plt.savefig('feature_diagnosis.png', dpi=150)
    plt.close()
    print("\nSaved diagnostic plot to feature_diagnosis.png")

    return X, y


# ── Step 2: Better feature extraction ──────────────────────────────

def extract_better_features():
    """Extract features using multiple pooling strategies and fc_norm."""
    from dataset import HVFDataset
    from octcube import OCTCubeRegression
    from main_octcube import TrainConfig
    from einops import rearrange
    import tqdm

    loader = torch.utils.data.DataLoader(
        HVFDataset(
            split_label="train",
            target_size=(TrainConfig.img_size, TrainConfig.img_size),
            normalize=True
        ),
        batch_size=1,
        num_workers=TrainConfig.num_workers,
    )

    model = OCTCubeRegression(
        img_size=TrainConfig.img_size,
        patch_size=TrainConfig.patch_size,
        num_frames=TrainConfig.num_frames,
        t_patch_size=TrainConfig.t_patch_size,
        size=TrainConfig.model_size,
        freeze_encoder=True,
        checkpoint_path=TrainConfig.checkpoint_path,
    ).cuda()
    model.eval()

    # Access the underlying ViT for fc_norm
    vit = model.encoder.model

    all_feats = {
        'mean_raw': [],       # Current approach (no fc_norm)
        'mean_normed': [],    # Mean pool + fc_norm (the correct way)
        'cls_token': [],      # CLS token output
        'max_pool': [],       # Max pool + fc_norm
        'mean_max': [],       # Concat mean+max (2048-dim)
        'attn_pool': [],      # AttentionPool from the trained head
        'std_pool': [],       # Std across tokens (captures variance = pathology spread)
    }
    all_labels = []
    all_mrns = []

    with torch.no_grad():
        for batch in tqdm.tqdm(loader, desc="Extracting features"):
            x = batch['frames'].cuda().half()

            # Rearrange (B, C, T, H, W) -> (B, T, C, H, W) for encoder
            x = rearrange(x, "B C T H W -> B T C H W")

            # Get all tokens from encoder, but BEFORE the final norm + pool
            # We need to go through forward_features manually
            B, T, C, H, W = x.shape

            # Patch embed
            x_enc = vit.patch_embed(x)
            _, T_patches, L, D = x_enc.shape

            # Positional embeddings
            if vit.sep_pos_embed:
                x_enc = x_enc + vit.pos_embed_spatial.unsqueeze(1)
                x_enc = x_enc + vit.pos_embed_temporal.unsqueeze(2)

            x_enc = rearrange(x_enc, 'b t l d -> b (t l) d')

            if vit.cls_embed:
                cls_tokens = vit.cls_token.expand(B, -1, -1)
                x_enc = torch.cat([cls_tokens, x_enc], dim=1)

            x_enc = vit.pos_drop(x_enc)

            for blk in vit.blocks:
                x_enc = blk(x_enc)

            x_enc = vit.norm(x_enc)

            # Now x_enc is (B, 1+T*L, D) with cls token at position 0
            cls_out = x_enc[:, 0]                    # (B, D)
            patch_tokens = x_enc[:, 1:]               # (B, T*L, D)

            # Different pooling strategies
            mean_raw = patch_tokens.mean(dim=1).float()
            mean_normed = vit.fc_norm(patch_tokens.mean(dim=1)).float()
            max_pool = vit.fc_norm(patch_tokens.max(dim=1).values).float()
            mean_max = torch.cat([mean_normed, max_pool], dim=-1).float()
            std_pool = patch_tokens.float().std(dim=1)

            # Attention pool from the regression head
            attn_pool = model.head.pool(patch_tokens).float()

            all_feats['mean_raw'].append(mean_raw.cpu())
            all_feats['mean_normed'].append(mean_normed.cpu())
            all_feats['cls_token'].append(cls_out.float().cpu())
            all_feats['max_pool'].append(max_pool.cpu())
            all_feats['mean_max'].append(mean_max.cpu())
            all_feats['attn_pool'].append(attn_pool.cpu())
            all_feats['std_pool'].append(std_pool.cpu())
            all_labels.append(batch['label'])
            all_mrns.append(batch['mrn'])

    labels = torch.cat(all_labels).numpy()
    np.save('labels_v2.npy', labels)

    results = {}
    for name, feat_list in all_feats.items():
        feat_arr = torch.cat(feat_list).numpy()
        np.save(f'features_{name}.npy', feat_arr)
        results[name] = feat_arr
        print(f"Saved features_{name}.npy — shape {feat_arr.shape}")

    return results, labels


# ── Step 3: Compare pooling strategies ──────────────────────────────

def compare_pooling_strategies(results=None, labels=None):
    """Train Ridge regression on each feature variant and compare."""
    from sklearn.linear_model import Ridge, Lasso
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    if results is None:
        # Load from saved files
        results = {}
        labels = np.load('labels_v2.npy') if Path('labels_v2.npy').exists() else np.load('labels.npy')
        for name in ['mean_raw', 'mean_normed', 'cls_token', 'max_pool', 'mean_max', 'attn_pool', 'std_pool']:
            p = Path(f'features_{name}.npy')
            if p.exists():
                results[name] = np.load(p)

    if not results:
        print("No feature files found. Run extract_better_features() first.")
        return

    dummy_mae = np.abs(labels - labels.mean()).mean()
    print(f"\nDummy (predict mean) MAE: {dummy_mae:.4f}")
    print(f"Label std: {labels.std():.4f}")
    print(f"{'─'*70}")
    print(f"{'Feature variant':<20} {'dim':>5} {'Ridge MAE':>12} {'Improvement':>12} {'GBT MAE':>12}")
    print(f"{'─'*70}")

    for name, X in sorted(results.items()):
        pipe = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        ridge_scores = cross_val_score(pipe, X, labels, cv=5, scoring='neg_mean_absolute_error')
        ridge_mae = -ridge_scores.mean()

        pipe_gbt = make_pipeline(
            StandardScaler(),
            GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
        )
        gbt_scores = cross_val_score(pipe_gbt, X, labels, cv=5, scoring='neg_mean_absolute_error')
        gbt_mae = -gbt_scores.mean()

        improvement = (1 - ridge_mae / dummy_mae) * 100
        print(f"{name:<20} {X.shape[1]:>5} {ridge_mae:>12.4f} {improvement:>+11.1f}% {gbt_mae:>12.4f}")

    print(f"{'─'*70}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "extract":
        # Step 2: Extract better features (requires GPU + data)
        results, labels = extract_better_features()
        compare_pooling_strategies(results, labels)
    elif len(sys.argv) > 1 and sys.argv[1] == "compare":
        # Step 3: Compare already-extracted features
        compare_pooling_strategies()
    else:
        # Step 1: Diagnose existing features (no GPU needed)
        print("=" * 60)
        print("DIAGNOSING EXISTING FEATURES")
        print("=" * 60)
        diagnose_existing_features()
        print("\n" + "=" * 60)
        print("To extract better features:  python diagnose.py extract")
        print("To compare features:         python diagnose.py compare")
        print("=" * 60)
