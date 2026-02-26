"""
Training script for OCTCube-based HVF MTD prediction.

Two-phase training: head on pre-extracted features, then LoRA fine-tuning.
"""

import torch
import torch.nn as nn
from torch.optim import Optimizer
import torch.nn.functional as F
from torch.amp.grad_scaler import GradScaler
from torch.distributions import Beta
import numpy as np
from dataset import HVFDataset, FeatureDataset
import matplotlib.pyplot as plt
import time
import warnings
from jaxtyping import Float, jaxtyped
from beartype import beartype
from torch import Tensor
from dataclasses import dataclass
from typing import Optional
import os
import shutil
from collections import deque
import tqdm

from octcube import OCTCubeRegression, BetaRegressionHead, beta_nll_loss, apply_lora_to_encoder

typechecked = jaxtyped(typechecker=beartype)


@dataclass
class TrainConfig:
    # Training parameters
    step_size: int = 48  # Number of slices per volume chunk (must be divisible by t_patch_size=3)
    partial_val_interval: int = 1000  # Validation every N training steps in phase 2
    train_save_im: int = 30
    plot_losses: int = 10
    val_max_volumes: int = 100  # Number of volumes for partial validation
    scatter_dir: str = "scatter_plots"  # Directory for per-epoch heatmap scatters

    # Two-phase training
    phase1_epochs: int = 5    # Head-only (encoder frozen)
    phase2_epochs: int = 15   # End-to-end (encoder unfrozen with warmup)
    encoder_lr: float = 1e-5
    head_lr: float = 1e-3
    phase2_head_lr: float = 1e-4  # Lower LR for head MLP in phase 2 (already pretrained)
    phase2_pool_lr: float = 1e-4  # Pool LR — must stay close to MLP LR to avoid disrupting learned features
    warmup_fraction: float = 0.1  # Fraction of phase 2 steps for encoder LR warmup
    grad_accum_steps: int = 4  # Gradient accumulation steps (effective batch = batch_size * accum)

    # LoRA config for phase 2
    lora_rank: int = 16
    lora_alpha: float = 16.0

    # Pre-extracted features for phase 1 (skip encoder forward pass entirely)
    feature_dir: str = "extracted_features"
    feature_pool: str = "mean_raw"  # Must match AttentionPool init (mean pool at start)

    # Model parameters
    img_size: int = 384         # Reduced from 512: 24x24=576 spatial patches (vs 32x32=1024)
    patch_size: int = 16
    num_frames: int = 48
    t_patch_size: int = 3
    model_size: str = 'large'
    center_crop_frac: float = 0.5  # Crop to center 50% of W before resize (None to disable)
    phase1_batch_size: int = 2   # Also used for LoRA phase 2 (similar memory footprint)
    phase2_batch_size: int = 1   # Only needed if doing full fine-tuning (no LoRA)
    num_workers: int = 25

    # Paths
    checkpoint_path: Optional[str] = '/storage2/fs1/leeay/Active/jstrand/projects/OCTCubeM/ckpt/OCTCube.pth'  # Path to OCTCube pretrained encoder weights
    save_dir: str = "checkpoints_octcube"
    resume_from: Optional[str] = None  # Path to resume training from (latest checkpoint)


def _format_time(seconds):
    """Format seconds as human-readable duration."""
    seconds = int(seconds)
    if seconds >= 3600:
        return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"
    return f"{seconds // 60}m{seconds % 60:02d}s"


class Metrics:
    def __init__(self):
        from collections import defaultdict

        self.data = {}
        for split in ["train", "val", "val_partial", "test"]:
            self.data[split] = {"iterations": [], "metrics": defaultdict(list)}
        self.current_iter = 0
        self.current_epoch = 0
        self.rolling_preds = deque(maxlen=100)
        self.rolling_gts = deque(maxlen=100)
        self.eta_str = ""

    def append(self, split, metrics):
        if split == "train":
            self.current_iter += 1

        self.data[split]["iterations"].append(self.current_iter)
        for k, v in metrics.items():
            self.data[split]["metrics"][k].append(v)
        self.print_latest(splits=split)

    def append_regression(self, preds, targets):
        """Call each step with predictions and GT."""
        self.rolling_preds.append(preds.detach().cpu())
        self.rolling_gts.append(targets.cpu())

    def get_regression_metrics(self):
        preds = torch.cat(list(self.rolling_preds))
        gts = torch.cat(list(self.rolling_gts))
        mae = (preds - gts).abs().mean().item()
        r = torch.corrcoef(torch.stack([preds, gts]))[0, 1].item()

        # R² = 1 - SS_res / SS_tot
        ss_res = ((preds - gts) ** 2).sum().item()
        ss_tot = ((gts - gts.mean()) ** 2).sum().item()
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        return {"mae": mae, "pearson_r": r, "r2": r2}

    def print_latest(self, splits=None):
        if splits is None:
            splits = ["train", "val", "val_partial", "test"]
        elif isinstance(splits, str):
            splits = [splits]

        for split in splits:
            if len(self.data[split]["iterations"]) == 0:
                continue

            latest_iter = self.data[split]["iterations"][-1]
            metrics_str = ", ".join(
                f"{k}={v[-1]:.4f}"
                for k, v in self.data[split]["metrics"].items()
                if len(v) > 0
            )
            eta_part = f" | {self.eta_str}" if self.eta_str and split == "train" else ""
            print(f"{split} [{self.current_epoch}:{latest_iter}]: {metrics_str}{eta_part}")

    def plot_metrics(self):
        print("Plotting Metrics")
        all_metrics = set()
        for split in self.data.values():
            all_metrics.update(split["metrics"].keys())

        if len(all_metrics) == 0:
            return

        fig, axes = plt.subplots(len(all_metrics), 1, figsize=(10, 4 * len(all_metrics)))
        if len(all_metrics) == 1:
            axes = [axes]

        for ax, metric_name in zip(axes, sorted(all_metrics)):
            for split in ["train", "val", "val_partial", "test"]:
                iters = self.data[split]["iterations"]
                vals = self.data[split]["metrics"][metric_name]
                if len(vals) > 0:
                    if split == "train":
                        ax.plot(iters, vals, alpha=0.7, label=split)
                    else:
                        ax.scatter(iters, vals, label=split, s=50, zorder=5)
            ax.set_ylabel(metric_name)
            if metric_name == "loss":
                ax.set_yscale("log")
                ax.set_ylim(top=1e3)
            elif metric_name == "r2":
                ax.set_ylim(0, 1)
            ax.legend()
        axes[-1].set_xlabel("iteration")
        plt.tight_layout()
        plt.savefig("octcube_metrics.png")
        plt.close()

    def save(self, path):
        import json
        save_data = {
            "current_iter": self.current_iter,
            "current_epoch": self.current_epoch,
            "data": {
                split: {
                    "iterations": self.data[split]["iterations"],
                    "metrics": dict(self.data[split]["metrics"]),
                }
                for split in self.data
            },
        }
        with open(path, "w") as f:
            json.dump(save_data, f)

    def load(self, path):
        import json
        from collections import defaultdict
        with open(path, "r") as f:
            save_data = json.load(f)
        self.current_iter = save_data["current_iter"]
        self.current_epoch = save_data.get("current_epoch", 0)
        for split in save_data["data"]:
            self.data[split]["iterations"] = save_data["data"][split]["iterations"]
            self.data[split]["metrics"] = defaultdict(list, save_data["data"][split]["metrics"])

    def should_save_train_images(self):
        return self.current_iter % TrainConfig.train_save_im == 0

    def should_plot_losses(self):
        return self.current_iter % TrainConfig.plot_losses == 0


def plot_pred_vs_gt_heatmap(preds_np, gts_np, epoch, save_dir="scatter_plots", title=""):
    """Save a 2D histogram heatmap of predictions vs ground truth."""
    os.makedirs(save_dir, exist_ok=True)

    r2 = 1.0 - np.sum((preds_np - gts_np) ** 2) / max(np.sum((gts_np - gts_np.mean()) ** 2), 1e-8)
    mae = np.abs(preds_np - gts_np).mean()

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    vmin = min(gts_np.min(), preds_np.min())
    vmax = max(gts_np.max(), preds_np.max())
    bins = np.linspace(vmin, vmax, 60)
    h, xedges, yedges = np.histogram2d(gts_np, preds_np, bins=bins)
    h_log = np.log1p(h)
    ax.imshow(h_log.T, origin='lower', aspect='auto',
              extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
              cmap='viridis')
    ax.plot([vmin, vmax], [vmin, vmax], 'r--', linewidth=1.5, label='y=x')
    ax.set_xlabel('Ground Truth')
    ax.set_ylabel('Predicted')
    title_str = f'Epoch {epoch}'
    if title:
        title_str = f'{title} - {title_str}'
    ax.set_title(f'{title_str}\nR\u00b2={r2:.4f}  MAE={mae:.4f}')
    ax.legend(loc='upper left', fontsize=8)
    plt.tight_layout()
    path = os.path.join(save_dir, f"epoch_{epoch:03d}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved heatmap scatter to {path}")


def save_checkpoint(model, optimizer, scaler, metrics, path, is_best=False, save_encoder=False):
    """Save training checkpoint. Optionally saves encoder state for phase 2."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "head_state_dict": model.head.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "metrics": {
            "current_iter": metrics.current_iter,
            "current_epoch": metrics.current_epoch,
            "data": {
                split: {
                    "iterations": metrics.data[split]["iterations"],
                    "metrics": dict(metrics.data[split]["metrics"]),
                }
                for split in metrics.data
            },
        },
        "is_best": is_best,
    }
    if save_encoder:
        checkpoint["encoder_state_dict"] = model.encoder.state_dict()
    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")


def load_checkpoint(model, optimizer, scaler, metrics, path):
    """Load training checkpoint. Restores encoder state if present."""
    from collections import defaultdict

    if not os.path.exists(path):
        print(f"No checkpoint found at {path}")
        return False

    checkpoint = torch.load(path, map_location="cuda", weights_only=False)

    model.head.load_state_dict(checkpoint["head_state_dict"])
    if "encoder_state_dict" in checkpoint:
        model.encoder.load_state_dict(checkpoint["encoder_state_dict"])
        print("  Loaded encoder state from checkpoint")
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])

    # Restore metrics
    metrics_data = checkpoint["metrics"]
    metrics.current_iter = metrics_data["current_iter"]
    metrics.current_epoch = metrics_data["current_epoch"]
    for split in metrics_data["data"]:
        metrics.data[split]["iterations"] = metrics_data["data"][split]["iterations"]
        metrics.data[split]["metrics"] = defaultdict(
            list, metrics_data["data"][split]["metrics"]
        )

    print(f"Loaded checkpoint from {path} (epoch {metrics.current_epoch}, iter {metrics.current_iter})")
    return True

def get_warmup_scheduler(optimizer, warmup_steps):
    """Linear warmup for encoder/pool (groups 0-1), constant for head MLP (group 2).

    Three param groups expected:
      0 = LoRA adapter params  (warmup from 0)
      1 = AttentionPool params (warmup from 0 — random init, never saw data in phase 1)
      2 = Head MLP params      (constant — already pretrained in phase 1)
    """
    def lr_lambda_warmup(step):
        if warmup_steps == 0:
            return 1.0
        if step < warmup_steps:
            return step / warmup_steps
        return 1.0

    def lr_lambda_constant(step):
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, [lr_lambda_warmup, lr_lambda_warmup, lr_lambda_constant]
    )


def train_one_epoch(
    model,
    optimizer,
    scaler,
    train_dataloader,
    val_dataloader,
    metrics: Metrics,
    scheduler=None,
    remaining_epochs: int = 1,
):
    """Training epoch with gradient accumulation."""
    model.train()
    accum = TrainConfig.grad_accum_steps
    total_batches = len(train_dataloader)
    epoch_start = time.time()

    optimizer.zero_grad()

    outlier_log_path = os.path.join(TrainConfig.save_dir, "loss_outliers.jsonl")
    NLL_OUTLIER_THRESHOLD = 5.0  # Log samples with raw NLL above this

    for batch_idx, batch in enumerate(train_dataloader):
        imgs = batch['frames']
        labels = batch['label']
        mrns = batch['mrn']

        step_metrics, preds, alpha, beta_param, raw_nll = one_training_step(
            imgs,
            labels,
            model,
            scaler,
            accum,
        )

        # Log outlier samples for investigation
        for i in range(len(raw_nll)):
            if raw_nll[i].item() > NLL_OUTLIER_THRESHOLD:
                import json
                entry = {
                    "epoch": metrics.current_epoch,
                    "iter": metrics.current_iter,
                    "mrn": str(mrns[i]),
                    "label_normalized": labels[i].item(),
                    "alpha": alpha[i].item(),
                    "beta": beta_param[i].item(),
                    "predicted_mean": (alpha[i] / (alpha[i] + beta_param[i])).item(),
                    "raw_nll": raw_nll[i].item(),
                }
                with open(outlier_log_path, "a") as f:
                    f.write(json.dumps(entry) + "\n")

        # Optimizer step on accumulation boundary
        if (batch_idx + 1) % accum == 0 or (batch_idx + 1) == total_batches:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

        # ETA calculation
        elapsed = time.time() - epoch_start
        steps_done = batch_idx + 1
        secs_per_step = elapsed / steps_done
        epoch_remaining = secs_per_step * (total_batches - steps_done)
        total_remaining = epoch_remaining + secs_per_step * total_batches * (remaining_epochs - 1)
        metrics.eta_str = (
            f"batch {steps_done}/{total_batches} | "
            f"{_format_time(elapsed)} elapsed, ETA {_format_time(total_remaining)}"
        )

        metrics.append_regression(preds, labels)
        step_metrics.update(metrics.get_regression_metrics())
        metrics.append("train", step_metrics)

        if metrics.should_plot_losses():
            metrics.plot_metrics()

        # Periodic validation
        if metrics.current_iter % TrainConfig.partial_val_interval == 0:
            validation_partial_epoch(model, val_dataloader, metrics)


@typechecked
def one_training_step(
    images: Float[Tensor, "batches channels frames height width"],
    labels: Float[Tensor, "batches"],
    model: nn.Module,
    scaler: GradScaler,
    accum_steps: int,
) -> tuple[dict[str, float], Tensor, Float[Tensor, "batches"], Float[Tensor, "batches"], Float[Tensor, "batches"]]:
    """Single micro-step: forward + scaled backward (no optimizer step).
    Returns (metrics, preds, alpha, beta, raw_nll) for diagnostics."""
    with torch.amp.autocast("cuda"):
        logits = model(images.cuda())
        alpha = logits[0].float()
        beta = logits[1].float()

    # Beta.log_prob uses lgamma/log/pow — must be fp32 (fp16 causes spikes)
    loss, per_sample_nll = beta_nll_loss(alpha, beta, labels.cuda())
    scaled_loss = loss / accum_steps

    scaler.scale(scaled_loss).backward()

    with torch.no_grad():
        pred = Beta(alpha, beta).mean

    result = {"loss": np.exp(loss.item())}
    return result, pred, alpha.detach(), beta.detach(), per_sample_nll


def validation_partial_epoch(model, dataloader, metrics: Metrics, split="val_partial", max_vols=None):
    """Partial validation for regression: run ~max_vols volumes, compute metrics."""
    if max_vols is None:
        max_vols = TrainConfig.val_max_volumes

    print(f"Running partial validation ({max_vols} volumes)...")
    model.eval()

    all_preds = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0
    num_vols = 0

    with torch.no_grad():
        for batch in dataloader:
            if num_vols >= max_vols:
                break

            imgs = batch['frames'].cuda()
            labels = batch['label'].cuda()

            with torch.amp.autocast("cuda"):
                logits = model(imgs)
                alpha = logits[0].float()
                beta_val = logits[1].float()

            loss, _ = beta_nll_loss(alpha, beta_val, labels)
            pred = Beta(alpha, beta_val).mean

            total_loss += loss.item()
            all_preds.append(pred.cpu())
            all_labels.append(batch['label'])
            num_batches += 1
            num_vols += imgs.shape[0]

    preds = torch.cat(all_preds)
    gts = torch.cat(all_labels)

    avg_loss = total_loss / max(num_batches, 1)
    mae = (preds - gts).abs().mean().item()
    r = torch.corrcoef(torch.stack([preds, gts]))[0, 1].item()
    ss_res = ((preds - gts) ** 2).sum().item()
    ss_tot = ((gts - gts.mean()) ** 2).sum().item()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    val_metrics = {
        "loss": np.exp(avg_loss),
        "mae": mae,
        "pearson_r": r,
        "r2": r2,
    }

    metrics.append(split, val_metrics)
    metrics.plot_metrics()
    metrics.save("octcube_metrics.json")
    model.train()

    return preds.numpy(), gts.numpy()


def make_loaders(batch_size):
    """Create train/val DataLoaders with the given batch size."""
    dataset_kwargs = dict(
        target_size=(TrainConfig.img_size, TrainConfig.img_size),
        normalize=True,
        center_crop_frac=TrainConfig.center_crop_frac,
    )
    train_loader = torch.utils.data.DataLoader(
        HVFDataset(split_label="train", **dataset_kwargs),
        batch_size=batch_size,
        shuffle=True,
        num_workers=TrainConfig.num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        HVFDataset(split_label="val", **dataset_kwargs),
        batch_size=batch_size,
        num_workers=TrainConfig.num_workers,
    )
    return train_loader, val_loader


def phase1_on_features(metrics):
    """Phase 1: Train head MLP on pre-extracted features. No encoder needed."""
    fdir = TrainConfig.feature_dir
    pool = TrainConfig.feature_pool

    train_loader = torch.utils.data.DataLoader(
        FeatureDataset(f"{fdir}/{pool}_train.npy", f"{fdir}/labels_train.npy"),
        batch_size=256, shuffle=True, num_workers=4,
    )
    val_loader = torch.utils.data.DataLoader(
        FeatureDataset(f"{fdir}/{pool}_val.npy", f"{fdir}/labels_val.npy"),
        batch_size=256, num_workers=4,
    )

    embed_dim = 1024 if TrainConfig.model_size == 'large' else 768
    head = BetaRegressionHead(in_features=embed_dim, hidden_dim=256).cuda()

    optimizer = torch.optim.AdamW(head.parameters(), lr=TrainConfig.head_lr)

    print("=" * 60)
    print(f"Phase 1: Training head on pre-extracted features ({pool})")
    print(f"  {len(train_loader.dataset)} train, {len(val_loader.dataset)} val samples")
    print(f"  batch_size=256, no encoder forward pass needed")
    print("=" * 60)

    for e in range(1, TrainConfig.phase1_epochs + 1):
        metrics.current_epoch = e
        head.train()
        epoch_start = time.time()
        total_batches = len(train_loader)

        for batch_idx, batch in enumerate(train_loader):
            feats = batch["features"].cuda()
            labels = batch["label"].cuda()

            alpha, beta = head(feats)  # (B, D) -> head skips AttentionPool
            loss, _ = beta_nll_loss(alpha, beta, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = Beta(alpha, beta).mean
            metrics.append_regression(pred, batch["label"])
            step_metrics = {"loss": np.exp(loss.item())}
            step_metrics.update(metrics.get_regression_metrics())

            # ETA
            elapsed = time.time() - epoch_start
            steps_done = batch_idx + 1
            remaining_epochs = TrainConfig.phase1_epochs - e + 1
            secs_per_step = elapsed / steps_done
            epoch_remaining = secs_per_step * (total_batches - steps_done)
            total_remaining = epoch_remaining + secs_per_step * total_batches * (remaining_epochs - 1)
            metrics.eta_str = (
                f"batch {steps_done}/{total_batches} | "
                f"{_format_time(elapsed)} elapsed, ETA {_format_time(total_remaining)}"
            )

            metrics.append("train", step_metrics)

        # End-of-epoch: validate on feature val set
        head.eval()
        all_preds, all_labels = [], []
        total_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                feats = batch["features"].cuda()
                labels = batch["label"].cuda()
                alpha, beta = head(feats)
                loss, _ = beta_nll_loss(alpha, beta, labels)
                pred = Beta(alpha, beta).mean
                total_loss += loss.item()
                all_preds.append(pred.cpu())
                all_labels.append(batch["label"])

        preds = torch.cat(all_preds)
        gts = torch.cat(all_labels)
        avg_loss = total_loss / max(len(val_loader), 1)
        mae = (preds - gts).abs().mean().item()
        r = torch.corrcoef(torch.stack([preds, gts]))[0, 1].item()
        ss_res = ((preds - gts) ** 2).sum().item()
        ss_tot = ((gts - gts.mean()) ** 2).sum().item()
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        val_metrics = {"loss": np.exp(avg_loss), "mae": mae, "pearson_r": r, "r2": r2}
        metrics.append("val", val_metrics)

        # Heatmap scatter for this epoch
        plot_pred_vs_gt_heatmap(preds.numpy(), gts.numpy(), e, TrainConfig.scatter_dir, "Phase 1")

        metrics.plot_metrics()
        metrics.eta_str = ""
        print(f"  Epoch {e}/{TrainConfig.phase1_epochs} done")

    print("Phase 1 complete.")
    return head.state_dict()


def full_supervised_run():
    """Two-phase training: head on features, then LoRA fine-tuning on full model."""
    os.makedirs("selected_images", exist_ok=True)
    os.makedirs(TrainConfig.save_dir, exist_ok=True)
    os.makedirs(TrainConfig.scatter_dir, exist_ok=True)

    latest_path = os.path.join(TrainConfig.save_dir, "latest.pt")

    # Single metrics object shared across both phases
    metrics = Metrics()

    # ----------------------------------------------------------------
    # Phase 1: Train head on pre-extracted features (seconds, not hours)
    # ----------------------------------------------------------------
    head_state = phase1_on_features(metrics)

    # Save phase 1 head weights
    phase1_path = os.path.join(TrainConfig.save_dir, "phase1_head.pt")
    torch.save(head_state, phase1_path)
    print(f"Phase 1 head saved to {phase1_path}")

    # ----------------------------------------------------------------
    # Phase 2: LoRA fine-tuning (full model with encoder + adapted head)
    # ----------------------------------------------------------------
    print("=" * 60)
    print(f"Phase 2: LoRA fine-tuning for {TrainConfig.phase2_epochs} epochs")
    print(f"  LoRA rank={TrainConfig.lora_rank}, alpha={TrainConfig.lora_alpha}")
    print(f"  Encoder LR warmup over first {TrainConfig.warmup_fraction:.0%} of steps")
    spatial_patches = (TrainConfig.img_size // TrainConfig.patch_size) ** 2
    temporal_patches = TrainConfig.num_frames // TrainConfig.t_patch_size
    total_tokens = spatial_patches * temporal_patches
    print(f"  Tokens: {spatial_patches} spatial x {temporal_patches} temporal = {total_tokens:,}")
    if TrainConfig.center_crop_frac is not None:
        print(f"  Center crop: {TrainConfig.center_crop_frac:.0%} of W before resize")
    print("=" * 60)

    # Now create the full model (encoder + head)
    model = OCTCubeRegression(
        img_size=TrainConfig.img_size,
        patch_size=TrainConfig.patch_size,
        num_frames=TrainConfig.num_frames,
        t_patch_size=TrainConfig.t_patch_size,
        size=TrainConfig.model_size,
        freeze_encoder=True,
        checkpoint_path=TrainConfig.checkpoint_path,
    ).cuda()

    # Load phase 1 head weights into the full model's head
    model.head.load_state_dict(head_state)
    print("Loaded phase 1 head weights into full model")

    # Inject LoRA adapters into encoder attention layers
    lora_params = apply_lora_to_encoder(
        model.encoder,
        rank=TrainConfig.lora_rank,
        alpha=TrainConfig.lora_alpha,
    )
    print(f"LoRA injected: {lora_params:,} trainable params (vs ~300M frozen encoder)")

    scaler = GradScaler("cuda")
    train_loader, val_loader = make_loaders(TrainConfig.phase1_batch_size)

    # Collect only trainable params: LoRA adapters + pool + head MLP
    # AttentionPool was never trained in phase 1 (features were pre-pooled),
    # so it needs warmup just like the LoRA adapters.
    lora_adapter_params = [p for p in model.encoder.parameters() if p.requires_grad]
    pool_params = list(model.head.pool.parameters())
    mlp_params = list(model.head.mlp.parameters())
    optimizer = torch.optim.AdamW([
        {"params": lora_adapter_params, "lr": TrainConfig.encoder_lr},
        {"params": pool_params, "lr": TrainConfig.phase2_pool_lr},
        {"params": mlp_params, "lr": TrainConfig.phase2_head_lr},
    ])

    # Warmup scheduler for LoRA adapter LR
    # Scheduler steps once per optimizer step (every grad_accum_steps batches)
    total_optimizer_steps = (TrainConfig.phase2_epochs * len(train_loader)) // TrainConfig.grad_accum_steps
    warmup_steps = int(TrainConfig.warmup_fraction * total_optimizer_steps)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # LambdaLR.__init__ calls step() before any optimizer.step()
        scheduler = get_warmup_scheduler(optimizer, warmup_steps)
    print(f"Warmup: {warmup_steps} optimizer steps out of {total_optimizer_steps} total")

    for e in range(1, TrainConfig.phase2_epochs + 1):
        epoch_num = TrainConfig.phase1_epochs + e
        metrics.current_epoch = epoch_num
        print(f"Phase 2 - Epoch {e}/{TrainConfig.phase2_epochs} (overall {epoch_num})")
        train_one_epoch(
            model, optimizer, scaler,
            train_loader, val_loader, metrics,
            scheduler=scheduler,
            remaining_epochs=TrainConfig.phase2_epochs - e + 1,
        )

        # End-of-epoch validation + heatmap scatter
        preds, gts = validation_partial_epoch(model, val_loader, metrics, split="val")
        plot_pred_vs_gt_heatmap(preds, gts, epoch_num, TrainConfig.scatter_dir, "Phase 2")

        save_checkpoint(model, optimizer, scaler, metrics, latest_path, save_encoder=True)

    # Save final
    metrics.save(os.path.join(TrainConfig.save_dir, "metrics_final.json"))
    metrics.plot_metrics()
    print("Training complete.")


if __name__ == "__main__":
    full_supervised_run()
