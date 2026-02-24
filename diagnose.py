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

def extract_better_features(max_volumes=500):
    """Extract features using multiple pooling strategies, per split.

    Saves to extracted_features/{strategy}_{split}.npy and
    extracted_features/labels_{split}.npy.

    Uses the encoder's built-in forward() which handles temporal pos embed
    interpolation for variable frame counts.
    """
    from dataset import HVFDataset
    from octcube import OCTCubeRegression
    from main_octcube import TrainConfig
    from einops import rearrange
    import torch.nn.functional as F
    import tqdm
    import os

    out_dir = Path("extracted_features")
    out_dir.mkdir(exist_ok=True)

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

    splits = ["train", "val", "test"]
    all_results = {}  # {strategy_name: np.array} for the last split (for compare)
    all_labels = None

    for split_name in splits:
        loader = torch.utils.data.DataLoader(
            HVFDataset(
                split_label=split_name,
                target_size=(TrainConfig.img_size, TrainConfig.img_size),
                normalize=True
            ),
            batch_size=1,
            num_workers=TrainConfig.num_workers,
        )

        feats_by_strategy = {
            'mean_raw': [],       # Mean pool, no normalization
            'mean_normed': [],    # Mean pool + LayerNorm
            'max_pool': [],       # Max pool + LayerNorm
            'mean_max': [],       # Concat mean+max (2048-dim)
            'std_pool': [],       # Std across tokens (captures pathology spread)
            'attn_pool': [],      # Learned attention pooling (from regression head)
        }
        labels = []

        with torch.no_grad():
            with torch.amp.autocast("cuda"):
                for i, batch in enumerate(tqdm.tqdm(loader, desc=f"{split_name}")):
                    if i >= max_volumes:
                        break

                    x = batch['frames'].cuda()
                    # (B, C, T, H, W) -> (B, T, C, H, W) for encoder
                    x = rearrange(x, "B C T H W -> B T C H W")

                    # Use encoder's forward which handles pos embed interpolation
                    tokens = model.encoder(x, return_all_tokens=True)  # (B, Tp, L, D)
                    patch_tokens = tokens.flatten(1, 2)  # (B, Tp*L, D)

                    # Pooling strategies
                    mean_raw = patch_tokens.mean(dim=1).float()
                    mean_normed = F.layer_norm(mean_raw, (mean_raw.shape[-1],))
                    max_vals = patch_tokens.max(dim=1).values.float()
                    max_pool = F.layer_norm(max_vals, (max_vals.shape[-1],))
                    mean_max = torch.cat([mean_normed, max_pool], dim=-1)
                    std_pool = patch_tokens.float().std(dim=1)
                    attn_pooled = model.head.pool(patch_tokens).float()

                    feats_by_strategy['mean_raw'].append(mean_raw.cpu())
                    feats_by_strategy['mean_normed'].append(mean_normed.cpu())
                    feats_by_strategy['max_pool'].append(max_pool.cpu())
                    feats_by_strategy['mean_max'].append(mean_max.cpu())
                    feats_by_strategy['std_pool'].append(std_pool.cpu())
                    feats_by_strategy['attn_pool'].append(attn_pooled.cpu())
                    labels.append(batch['label'])

        y = torch.cat(labels).numpy()
        np.save(out_dir / f"labels_{split_name}.npy", y)

        for name, feat_list in feats_by_strategy.items():
            arr = torch.cat(feat_list).numpy()
            np.save(out_dir / f"{name}_{split_name}.npy", arr)
            if split_name == "train":
                all_results[name] = arr
        if split_name == "train":
            all_labels = y

        print(f"{split_name}: {len(labels)} volumes extracted")

    print(f"\nFeature extraction complete! Saved to {out_dir}/")
    return all_results, all_labels


# ── Step 3: Compare pooling strategies ──────────────────────────────

def compare_pooling_strategies(results=None, labels=None):
    """Train Ridge/GBT on each feature variant, report R², and plot scatters."""
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import cross_val_score, cross_val_predict
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import r2_score

    if results is None:
        # Load from saved files (prefer extracted_features/ directory)
        results = {}
        out_dir = Path("extracted_features")
        strategies = ['mean_raw', 'mean_normed', 'max_pool', 'mean_max', 'std_pool', 'attn_pool']

        if out_dir.exists():
            labels_path = out_dir / "labels_train.npy"
            labels = np.load(labels_path) if labels_path.exists() else None
            for name in strategies:
                p = out_dir / f"{name}_train.npy"
                if p.exists():
                    results[name] = np.load(p)
        else:
            labels = np.load('labels.npy') if Path('labels.npy').exists() else None
            for name in strategies:
                p = Path(f'features_{name}.npy')
                if p.exists():
                    results[name] = np.load(p)

    if not results:
        print("No feature files found. Run extract_better_features() first.")
        return

    dummy_mae = np.abs(labels - labels.mean()).mean()
    print(f"\nDummy (predict mean) MAE: {dummy_mae:.4f}")
    print(f"Label std: {labels.std():.4f}")
    print(f"{'─'*90}")
    print(f"{'Feature variant':<20} {'dim':>5} {'Ridge MAE':>12} {'Improvement':>12} {'GBT MAE':>12} {'GBT R²':>10}")
    print(f"{'─'*90}")

    # Collect GBT cross-val predictions for scatter plots
    gbt_predictions = {}

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

        # Get out-of-fold predictions for R² and scatter plots
        gbt_preds = cross_val_predict(pipe_gbt, X, labels, cv=5)
        gbt_r2 = r2_score(labels, gbt_preds)
        gbt_predictions[name] = gbt_preds

        improvement = (1 - ridge_mae / dummy_mae) * 100
        print(f"{name:<20} {X.shape[1]:>5} {ridge_mae:>12.4f} {improvement:>+11.1f}% {gbt_mae:>12.4f} {gbt_r2:>10.4f}")

    print(f"{'─'*90}")

    # ── Scatter plots: GBT predicted vs GT for each strategy ──
    n_strategies = len(gbt_predictions)
    ncols = min(3, n_strategies)
    nrows = (n_strategies + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), squeeze=False)

    for idx, (name, preds) in enumerate(sorted(gbt_predictions.items())):
        ax = axes[idx // ncols][idx % ncols]
        r2 = r2_score(labels, preds)
        mae = np.abs(labels - preds).mean()

        ax.scatter(labels, preds, alpha=0.4, s=15, edgecolors='none')
        # Perfect prediction line
        lims = [min(labels.min(), preds.min()), max(labels.max(), preds.max())]
        ax.plot(lims, lims, 'r--', linewidth=1, label='y=x')
        ax.set_xlabel('Ground Truth')
        ax.set_ylabel('Predicted')
        ax.set_title(f'{name}\nMAE={mae:.4f}  R²={r2:.4f}')
        ax.set_aspect('equal', adjustable='box')
        ax.legend(loc='upper left', fontsize=8)

    # Hide unused subplots
    for idx in range(n_strategies, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    plt.tight_layout()
    plt.savefig('gbt_scatter_plots.png', dpi=150)
    plt.close()
    print(f"\nSaved GBT scatter plots to gbt_scatter_plots.png")


# ── Step 4: Train MLP on frozen features ────────────────────────────

def train_mlp_on_features(strategy='mean_raw', epochs=300, lr=1e-3, hidden_dims=(512, 256)):
    """Train a simple MLP on saved frozen features with MSE loss.

    Uses proper train/val/test splits already created by extract_better_features.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    out_dir = Path("extracted_features")

    # Load data
    X_train = np.load(out_dir / f"{strategy}_train.npy")
    y_train = np.load(out_dir / "labels_train.npy")
    X_val = np.load(out_dir / f"{strategy}_val.npy")
    y_val = np.load(out_dir / "labels_val.npy")
    X_test = np.load(out_dir / f"{strategy}_test.npy")
    y_test = np.load(out_dir / "labels_test.npy")

    print(f"Strategy: {strategy}")
    print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    print(f"Feature dim: {X_train.shape[1]}")
    print(f"Label range: [{y_train.min():.4f}, {y_train.max():.4f}]")

    # Standardize features (fit on train only)
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    # Torch datasets
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    # Build MLP
    in_dim = X_train.shape[1]
    layers = []
    prev = in_dim
    for h in hidden_dims:
        layers.extend([
            nn.Linear(prev, h),
            nn.BatchNorm1d(h),
            nn.ReLU(),
            nn.Dropout(0.2),
        ])
        prev = h
    layers.append(nn.Linear(prev, 1))
    model = nn.Sequential(*layers)

    print(f"\nMLP: {in_dim} -> {' -> '.join(str(h) for h in hidden_dims)} -> 1")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20
    )
    loss_fn = nn.MSELoss()

    # Training loop
    best_val_mae = float('inf')
    best_state = None
    history = {'train_loss': [], 'val_mae': [], 'val_r2': []}

    for epoch in range(epochs):
        # Train
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            pred = model(xb).squeeze(-1)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        epoch_loss /= len(train_ds)

        # Validate
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t).squeeze(-1)
            val_mae = (val_pred - y_val_t).abs().mean().item()
            ss_res = ((y_val_t - val_pred) ** 2).sum().item()
            ss_tot = ((y_val_t - y_val_t.mean()) ** 2).sum().item()
            val_r2 = 1 - ss_res / ss_tot

        scheduler.step(val_mae)

        history['train_loss'].append(epoch_loss)
        history['val_mae'].append(val_mae)
        history['val_r2'].append(val_r2)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 50 == 0 or epoch == 0:
            cur_lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch+1:3d}  loss={epoch_loss:.5f}  "
                  f"val_MAE={val_mae:.4f}  val_R²={val_r2:.4f}  lr={cur_lr:.1e}")

    # Load best and evaluate on test
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test_t).squeeze(-1).numpy()
        val_pred_best = model(X_val_t).squeeze(-1).numpy()

    test_mae = np.abs(y_test - test_pred).mean()
    ss_res = ((y_test - test_pred) ** 2).sum()
    ss_tot = ((y_test - y_test.mean()) ** 2).sum()
    test_r2 = 1 - ss_res / ss_tot

    dummy_mae_test = np.abs(y_test - y_train.mean()).mean()

    print(f"\n{'─'*50}")
    print(f"Best val MAE:      {best_val_mae:.4f}")
    print(f"Test MAE:          {test_mae:.4f}")
    print(f"Test R²:           {test_r2:.4f}")
    print(f"Dummy MAE (test):  {dummy_mae_test:.4f}")
    print(f"Improvement:       {(1 - test_mae / dummy_mae_test) * 100:+.1f}%")
    print(f"{'─'*50}")

    # ── Plots ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Training curves
    axes[0].plot(history['train_loss'], label='Train loss', alpha=0.7)
    axes[0].plot(history['val_mae'], label='Val MAE', alpha=0.7)
    axes[0].axhline(dummy_mae_test, color='gray', linestyle='--', label='Dummy MAE')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss / MAE')
    axes[0].set_title('Training curves')
    axes[0].legend()

    # Val scatter
    axes[1].scatter(y_val, val_pred_best, alpha=0.4, s=15, edgecolors='none')
    lims = [min(y_val.min(), val_pred_best.min()), max(y_val.max(), val_pred_best.max())]
    axes[1].plot(lims, lims, 'r--', linewidth=1)
    val_r2_best = 1 - ((y_val - val_pred_best)**2).sum() / ((y_val - y_val.mean())**2).sum()
    axes[1].set_title(f'Val: MAE={np.abs(y_val - val_pred_best).mean():.4f}  R²={val_r2_best:.4f}')
    axes[1].set_xlabel('Ground Truth')
    axes[1].set_ylabel('Predicted')
    axes[1].set_aspect('equal', adjustable='box')

    # Test scatter
    axes[2].scatter(y_test, test_pred, alpha=0.4, s=15, edgecolors='none')
    lims = [min(y_test.min(), test_pred.min()), max(y_test.max(), test_pred.max())]
    axes[2].plot(lims, lims, 'r--', linewidth=1)
    axes[2].set_title(f'Test: MAE={test_mae:.4f}  R²={test_r2:.4f}')
    axes[2].set_xlabel('Ground Truth')
    axes[2].set_ylabel('Predicted')
    axes[2].set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.savefig('mlp_frozen_results.png', dpi=150)
    plt.close()
    print(f"Saved plots to mlp_frozen_results.png")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "extract":
        # Step 2: Extract better features (requires GPU + data)
        results, labels = extract_better_features()
        compare_pooling_strategies(results, labels)
    elif len(sys.argv) > 1 and sys.argv[1] == "compare":
        # Step 3: Compare already-extracted features
        compare_pooling_strategies()
    elif len(sys.argv) > 1 and sys.argv[1] == "mlp":
        # Step 4: Train MLP on frozen features
        strategy = sys.argv[2] if len(sys.argv) > 2 else "mean_raw"
        train_mlp_on_features(strategy=strategy)
    else:
        # Step 1: Diagnose existing features (no GPU needed)
        print("=" * 60)
        print("DIAGNOSING EXISTING FEATURES")
        print("=" * 60)
        diagnose_existing_features()
        print("\n" + "=" * 60)
        print("To extract better features:  python diagnose.py extract")
        print("To compare features:         python diagnose.py compare")
        print("To train MLP on features:    python diagnose.py mlp [strategy]")
        print("=" * 60)
