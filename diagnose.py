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

def _gpu_worker(gpu_id, work_queue, dataset, batch_size, results_dict, lock):
    """Worker that pulls work from a shared queue until empty."""
    from octcube import OCTCubeRegression
    from main_octcube import TrainConfig
    from einops import rearrange
    import torch.nn.functional as F
    import time

    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)

    model = OCTCubeRegression(
        img_size=TrainConfig.img_size,
        patch_size=TrainConfig.patch_size,
        num_frames=TrainConfig.num_frames,
        t_patch_size=TrainConfig.t_patch_size,
        size=TrainConfig.model_size,
        freeze_encoder=True,
        checkpoint_path=TrainConfig.checkpoint_path,
    ).to(device)
    model.eval()

    strategies = ['mean_raw', 'max_pool', 'attn_pool']
    n_processed = 0
    t0 = time.time()

    with torch.no_grad(), torch.amp.autocast("cuda"):
        while True:
            # Grab a chunk of indices from the queue
            try:
                chunk_idx, indices = work_queue.get_nowait()
            except Exception:
                break  # Queue empty, we're done

            subset = torch.utils.data.Subset(dataset, indices)
            loader = torch.utils.data.DataLoader(
                subset,
                batch_size=batch_size,
                num_workers=min(4, batch_size * 2),
                pin_memory=True,
                persistent_workers=False,
            )

            chunk_feats = {name: [] for name in strategies}
            chunk_labels = []

            for batch in loader:
                x = batch['frames'].to(device, non_blocking=True)
                x = rearrange(x, "B C T H W -> B T C H W")

                tokens = model.encoder(x, return_all_tokens=True)
                patch_tokens = tokens.flatten(1, 2)

                mean_raw = patch_tokens.mean(dim=1).float()
                max_vals = patch_tokens.max(dim=1).values.float()
                max_pool = F.layer_norm(max_vals, (max_vals.shape[-1],))
                attn_pooled = model.head.pool(patch_tokens).float()

                chunk_feats['mean_raw'].append(mean_raw.cpu())
                chunk_feats['max_pool'].append(max_pool.cpu())
                chunk_feats['attn_pool'].append(attn_pooled.cpu())
                chunk_labels.append(batch['label'])

            # Store results for this chunk
            result = {name: torch.cat(v).numpy() for name, v in chunk_feats.items()}
            y = torch.cat(chunk_labels).numpy()
            n_processed += len(y)

            with lock:
                results_dict[chunk_idx] = (indices, result, y)
                total_done = sum(len(v[2]) for v in results_dict.values())
                print(f"  GPU{gpu_id}: chunk {chunk_idx} done ({len(y)} samples) | "
                      f"total: {total_done} samples done")

    elapsed = time.time() - t0
    if n_processed > 0:
        print(f"  GPU{gpu_id}: finished {n_processed} samples in {elapsed:.0f}s "
              f"({n_processed/elapsed:.1f} vol/s)")


def extract_better_features(max_volumes=None, batch_size=4, num_gpus=None, chunk_size=32):
    """Extract features across multiple GPUs with dynamic work stealing.

    Each GPU pulls chunks from a shared queue, so faster GPUs process more.

    Args:
        max_volumes: Max samples per split (None = all data).
        batch_size: Batch size per GPU. 4 fits comfortably in 40GB.
        num_gpus: Number of GPUs to use (None = auto-detect).
        chunk_size: Samples per work unit. Smaller = better load balance.
    """
    from dataset import HVFDataset
    from main_octcube import TrainConfig
    from concurrent.futures import ThreadPoolExecutor
    import queue
    import threading
    import time

    if num_gpus is None:
        num_gpus = torch.cuda.device_count()
    print(f"Using {num_gpus} GPUs, batch_size={batch_size}/GPU, chunk_size={chunk_size}")

    out_dir = Path("extracted_features")
    out_dir.mkdir(exist_ok=True)

    splits = ["train", "val", "test"]

    for split_name in splits:
        dataset = HVFDataset(
            split_label=split_name,
            target_size=(TrainConfig.img_size, TrainConfig.img_size),
            normalize=True
        )
        n_total = len(dataset)
        if max_volumes is not None:
            n_total = min(n_total, max_volumes)

        # Build work queue: each item is (chunk_id, list_of_indices)
        all_indices = list(range(n_total))
        work_queue = queue.Queue()
        n_chunks = 0
        for start in range(0, n_total, chunk_size):
            chunk = all_indices[start:start + chunk_size]
            work_queue.put((n_chunks, chunk))
            n_chunks += 1

        print(f"\n{split_name}: {n_total} samples in {n_chunks} chunks across {num_gpus} GPUs")

        results_dict = {}
        lock = threading.Lock()
        t0 = time.time()

        with ThreadPoolExecutor(max_workers=num_gpus) as pool:
            futures = []
            for gpu_id in range(num_gpus):
                fut = pool.submit(
                    _gpu_worker, gpu_id, work_queue, dataset,
                    batch_size, results_dict, lock,
                )
                futures.append(fut)
            # Wait for all to finish (and propagate exceptions)
            for fut in futures:
                fut.result()

        # Merge results in original order
        strategies = ['mean_raw', 'max_pool', 'attn_pool']
        merged_feats = {name: np.empty((n_total, 1024), dtype=np.float32) for name in strategies}
        merged_labels = np.empty(n_total, dtype=np.float32)

        for chunk_idx in sorted(results_dict.keys()):
            indices, result, y = results_dict[chunk_idx]
            for name in strategies:
                merged_feats[name][indices] = result[name]
            merged_labels[indices] = y

        np.save(out_dir / f"labels_{split_name}.npy", merged_labels)
        for name, arr in merged_feats.items():
            np.save(out_dir / f"{name}_{split_name}.npy", arr)

        elapsed = time.time() - t0
        rate = n_total / elapsed if elapsed > 0 else 0
        print(f"{split_name} total: {n_total} samples in {elapsed:.0f}s ({rate:.1f} vol/s)")

    print(f"\nFeature extraction complete! Saved to {out_dir}/")


def scaling_curve(strategies=None, n_points=6):
    """Plot R² vs training set size to see if performance is still climbing.

    Requires features already extracted (the full set). Subsamples the
    training data at geometrically spaced sizes and evaluates on the
    full test set each time.
    """
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import r2_score

    out_dir = Path("extracted_features")
    if strategies is None:
        strategies = ['mean_raw', 'max_pool', 'attn_pool']

    y_train_full = np.load(out_dir / "labels_train.npy")
    y_test = np.load(out_dir / "labels_test.npy")
    N = len(y_train_full)

    # Geometrically spaced sizes from 50 up to full N
    sizes = np.unique(np.geomspace(50, N, n_points).astype(int))
    sizes = [s for s in sizes if s <= N]

    print(f"Scaling curve: {len(sizes)} sizes from {sizes[0]} to {sizes[-1]}")
    print(f"Test set: {len(y_test)} samples")
    print()

    results = {}  # {strategy: {n: r2}}

    for strat in strategies:
        train_p = out_dir / f"{strat}_train.npy"
        test_p = out_dir / f"{strat}_test.npy"
        if not train_p.exists() or not test_p.exists():
            print(f"Skipping {strat} (files not found)")
            continue

        X_train_full = np.load(train_p)
        X_test = np.load(test_p)
        results[strat] = {}

        print(f"{strat}:")
        for n in sizes:
            # Subsample training set (deterministic: first n samples)
            X_sub = X_train_full[:n]
            y_sub = y_train_full[:n]

            pipe = make_pipeline(
                StandardScaler(),
                GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
            )
            pipe.fit(X_sub, y_sub)
            preds = pipe.predict(X_test)
            r2 = r2_score(y_test, preds)
            mae = np.abs(y_test - preds).mean()
            results[strat][n] = r2
            print(f"  N={n:>6d}  R²={r2:.4f}  MAE={mae:.4f}")

        print()

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    for strat, r2_by_n in results.items():
        ns = sorted(r2_by_n.keys())
        r2s = [r2_by_n[n] for n in ns]
        ax.plot(ns, r2s, 'o-', label=strat, markersize=6)

    ax.set_xlabel('Training set size (N)')
    ax.set_ylabel('Test R²')
    ax.set_title('Scaling curve: R² vs training data')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('scaling_curve.png', dpi=150)
    plt.close()
    print(f"Saved scaling_curve.png")


# ── Step 3: Compare pooling strategies ──────────────────────────────

def compare_pooling_strategies():
    """Train Ridge/GBT on train set, evaluate on held-out test set."""
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import r2_score

    out_dir = Path("extracted_features")
    if not out_dir.exists():
        print("extracted_features/ not found. Run extract_better_features() first.")
        return

    y_train = np.load(out_dir / "labels_train.npy")
    y_test = np.load(out_dir / "labels_test.npy")

    # Auto-detect available strategies from files
    train_data = {}
    test_data = {}
    for f in sorted(out_dir.glob("*_train.npy")):
        name = f.stem.replace("_train", "")
        if name == "labels":
            continue
        test_p = out_dir / f"{name}_test.npy"
        if test_p.exists():
            train_data[name] = np.load(f)
            test_data[name] = np.load(test_p)

    # Back-compat: also check legacy names
    for name in ['mean_raw', 'mean_normed', 'max_pool', 'mean_max', 'std_pool', 'attn_pool']:
        if name in train_data:
            continue
        train_p = out_dir / f"{name}_train.npy"
        test_p = out_dir / f"{name}_test.npy"
        if train_p.exists() and test_p.exists():
            train_data[name] = np.load(train_p)
            test_data[name] = np.load(test_p)

    if not train_data:
        print("No feature files found. Run extract_better_features() first.")
        return

    dummy_mae = np.abs(y_test - y_train.mean()).mean()
    print(f"\nDummy (predict mean) test MAE: {dummy_mae:.4f}")
    print(f"Train label std: {y_train.std():.4f}, Test label std: {y_test.std():.4f}")
    print(f"Train: {len(y_train)}, Test: {len(y_test)}")
    print(f"{'─'*100}")
    print(f"{'Feature variant':<20} {'dim':>5} {'Ridge MAE':>12} {'Ridge R²':>10} {'GBT MAE':>12} {'GBT R²':>10} {'Improvement':>12}")
    print(f"{'─'*100}")

    # Collect test predictions for scatter plots
    gbt_test_preds = {}

    for name in sorted(train_data.keys()):
        X_train = train_data[name]
        X_test = test_data[name]

        # Ridge: train on full train, predict on test
        pipe_ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        pipe_ridge.fit(X_train, y_train)
        ridge_preds = pipe_ridge.predict(X_test)
        ridge_mae = np.abs(y_test - ridge_preds).mean()
        ridge_r2 = r2_score(y_test, ridge_preds)

        # GBT: train on full train, predict on test
        pipe_gbt = make_pipeline(
            StandardScaler(),
            GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
        )
        pipe_gbt.fit(X_train, y_train)
        gbt_preds = pipe_gbt.predict(X_test)
        gbt_mae = np.abs(y_test - gbt_preds).mean()
        gbt_r2 = r2_score(y_test, gbt_preds)
        gbt_test_preds[name] = gbt_preds

        improvement = (1 - gbt_mae / dummy_mae) * 100
        print(f"{name:<20} {X_train.shape[1]:>5} {ridge_mae:>12.4f} {ridge_r2:>10.4f} {gbt_mae:>12.4f} {gbt_r2:>10.4f} {improvement:>+11.1f}%")

    print(f"{'─'*100}")

    # ── Scatter plots: GBT predicted vs GT on test set ──
    n_strategies = len(gbt_test_preds)
    ncols = min(3, n_strategies)
    nrows = (n_strategies + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), squeeze=False)

    for idx, (name, preds) in enumerate(sorted(gbt_test_preds.items())):
        ax = axes[idx // ncols][idx % ncols]
        r2 = r2_score(y_test, preds)
        mae = np.abs(y_test - preds).mean()

        ax.scatter(y_test, preds, alpha=0.4, s=15, edgecolors='none')
        lims = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
        ax.plot(lims, lims, 'r--', linewidth=1, label='y=x')
        ax.set_xlabel('Ground Truth')
        ax.set_ylabel('Predicted')
        ax.set_title(f'{name}\nTest MAE={mae:.4f}  Test R²={r2:.4f}')
        ax.set_aspect('equal', adjustable='box')
        ax.legend(loc='upper left', fontsize=8)

    for idx in range(n_strategies, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    plt.tight_layout()
    plt.savefig('gbt_scatter_plots.png', dpi=150)
    plt.close()
    print(f"\nSaved GBT test scatter plots to gbt_scatter_plots.png")


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


def overfit_test(strategy='mean_raw', n_samples=50, epochs=2000, lr=1e-3):
    """Can an MLP memorize a small subset? Tests if features carry any signal.

    No dropout, no weight decay, no early stopping — pure memorization attempt.
    If train loss doesn't approach zero, the features are effectively in the
    null space for this task.
    """
    import torch
    import torch.nn as nn

    out_dir = Path("extracted_features")
    X_train = np.load(out_dir / f"{strategy}_train.npy")
    y_train = np.load(out_dir / "labels_train.npy")

    # Take a small subset to make memorization easy
    n = min(n_samples, len(X_train))
    X = X_train[:n]
    y = y_train[:n]

    # Standardize
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    X = (X - mean) / std

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    # Deliberately overparameterized MLP, no regularization
    model = nn.Sequential(
        nn.Linear(X.shape[1], 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 1),
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Overfit test: {n} samples, {X.shape[1]}-dim features, {n_params:,} params")
    print(f"Label range: [{y.min():.4f}, {y.max():.4f}], std: {y.std():.4f}")
    print(f"Dummy MSE (predict mean): {((y - y.mean())**2).mean():.6f}")
    print()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    history = []
    for epoch in range(epochs):
        model.train()
        pred = model(X_t).squeeze(-1)
        loss = loss_fn(pred, y_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        mse = loss.item()
        history.append(mse)

        if (epoch + 1) % 200 == 0 or epoch == 0 or mse < 1e-6:
            mae = (pred.detach() - y_t).abs().mean().item()
            pred_std = pred.detach().std().item()
            print(f"  Epoch {epoch+1:4d}  MSE={mse:.6f}  MAE={mae:.4f}  pred_std={pred_std:.4f}")
            if mse < 1e-6:
                print("  -> Successfully memorized!")
                break

    final_mse = history[-1]
    dummy_mse = ((y - y.mean())**2).mean()
    print(f"\n{'─'*50}")
    print(f"Final train MSE:   {final_mse:.6f}")
    print(f"Dummy MSE:         {dummy_mse:.6f}")
    print(f"Ratio:             {final_mse / dummy_mse:.4f}")

    if final_mse < dummy_mse * 0.01:
        print("PASS: Model can memorize — features carry signal")
    elif final_mse < dummy_mse * 0.5:
        print("PARTIAL: Model learns something but can't fully memorize")
    else:
        print("FAIL: Model barely beats predicting the mean — features may be in null space")
    print(f"{'─'*50}")

    # Plot loss curve
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history, label='Train MSE')
    ax.axhline(dummy_mse, color='red', linestyle='--', label='Dummy MSE (predict mean)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE')
    ax.set_title(f'Overfit test: {n} samples, {strategy}')
    ax.set_yscale('log')
    ax.legend()
    plt.tight_layout()
    plt.savefig('overfit_test.png', dpi=150)
    plt.close()
    print(f"Saved loss curve to overfit_test.png")


if __name__ == "__main__":
    import sys

    cmd = sys.argv[1] if len(sys.argv) > 1 else None

    if cmd == "extract":
        # Extract features across multiple GPUs
        max_vol = int(sys.argv[2]) if len(sys.argv) > 2 else None
        bs = int(sys.argv[3]) if len(sys.argv) > 3 else 4
        n_gpus = int(sys.argv[4]) if len(sys.argv) > 4 else None
        extract_better_features(max_volumes=max_vol, batch_size=bs, num_gpus=n_gpus)
    elif cmd == "compare":
        compare_pooling_strategies()
    elif cmd == "scaling":
        scaling_curve()
    elif cmd == "mlp":
        strategy = sys.argv[2] if len(sys.argv) > 2 else "mean_raw"
        train_mlp_on_features(strategy=strategy)
    elif cmd == "overfit":
        strategy = sys.argv[2] if len(sys.argv) > 2 else "mean_raw"
        n = int(sys.argv[3]) if len(sys.argv) > 3 else 50
        overfit_test(strategy=strategy, n_samples=n)
    else:
        print("=" * 60)
        print("DIAGNOSING EXISTING FEATURES")
        print("=" * 60)
        diagnose_existing_features()
        print("\n" + "=" * 60)
        print("Usage:")
        print("  python diagnose.py extract [max_volumes] [batch_size] [num_gpus]")
        print("    Extract features across GPUs. Omit max_volumes for all data.")
        print("    Example: python diagnose.py extract           # all data, bs=4, all GPUs")
        print("    Example: python diagnose.py extract 5000      # first 5k per split")
        print("    Example: python diagnose.py extract None 4 4  # all data, bs=4, 4 GPUs")
        print()
        print("  python diagnose.py compare     # Ridge/GBT on extracted features")
        print("  python diagnose.py scaling     # R² vs N scaling curve")
        print("  python diagnose.py mlp [strategy]")
        print("  python diagnose.py overfit [strategy] [n_samples]")
        print("=" * 60)
