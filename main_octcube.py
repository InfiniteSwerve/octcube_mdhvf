"""
Training script for OCTCube-based segmentation.

This script adapts the MIRAGE training pipeline to work with OCTCube,
which processes 3D OCT volumes instead of individual slices.
"""

import torch
import torch.nn as nn
from torch.optim import Optimizer
import torch.nn.functional as F
from torch.amp.grad_scaler import GradScaler
from torch.distributions import Beta
import numpy as np
from dataset import HVFDataset
import matplotlib.pyplot as plt
import time
from jaxtyping import Float, jaxtyped
from beartype import beartype
from torch import Tensor
from dataclasses import dataclass
from typing import Optional
import os
import shutil
from collections import deque
import tqdm

from octcube import OCTCubeRegression, beta_nll_loss

typechecked = jaxtyped(typechecker=beartype)


@dataclass
class TrainConfig:
    # Training parameters
    step_size: int = 48  # Number of slices per volume chunk (must be divisible by t_patch_size=3)
    partial_val_interval: int = 500
    train_save_im: int = 30
    plot_losses: int = 30

    # Two-phase training
    phase1_epochs: int = 5    # Head-only (encoder frozen)
    phase2_epochs: int = 15   # End-to-end (encoder unfrozen with warmup)
    encoder_lr: float = 1e-5
    head_lr: float = 1e-3
    warmup_fraction: float = 0.1  # Fraction of phase 2 steps for encoder LR warmup

    # Model parameters
    img_size: int = 512
    patch_size: int = 16
    num_frames: int = 48
    t_patch_size: int = 3
    model_size: str = 'large'
    batch_size = 1
    num_workers = 25

    # Paths
    checkpoint_path: Optional[str] = '/storage2/fs1/leeay/Active/jstrand/projects/OCTCubeM/ckpt/OCTCube.pth'  # Path to OCTCube pretrained encoder weights
    save_dir: str = "checkpoints_octcube"
    resume_from: Optional[str] = None  # Path to resume training from (latest checkpoint)


class Metrics:
    def __init__(self):
        from collections import defaultdict

        self.data = {}
        for split in ["train", "val", "val_partial", "test"]:
            self.data[split] = {"iterations": [], "metrics": defaultdict(list)}
        self.current_iter = 0
        self.current_epoch = 0
        self.rolling_preds = deque(maxlen=100)
        self.rolling_gts= deque(maxlen=100)

    def append(self, split, metrics):
        if split == "train":
            self.current_iter += 1

        self.data[split]["iterations"].append(self.current_iter)
        for k, v in metrics.items():
            self.data[split]["metrics"][k].append(v)
        self.print_latest(splits=split)

    def append_regression(self, preds, targets):
        """Call each step with denormalized predictions and GT."""
        self.rolling_preds.append(preds.detach().cpu())
        self.rolling_gts.append(targets.cpu())

    def get_regression_metrics(self):
        preds = torch.cat(list(self.rolling_preds))
        gts = torch.cat(list(self.rolling_gts))
        mae = (preds - gts).abs().mean().item()
        r = torch.corrcoef(torch.stack([preds, gts]))[0, 1].item()
        preds_np = preds.numpy()
        gts_np = gts.numpy()
        lo = min(preds_np.min(), gts_np.min())
        hi = max(preds_np.max(), gts_np.max())
        plt.scatter(gts_np, preds_np, alpha=0.5)
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlabel('GT')
        plt.ylabel('Predicted')
        plt.savefig("regression_scatter.png")
        plt.close()


        return {"mae": mae, "pearson_r": r}

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
            print(f"{split} [{self.current_epoch}:{latest_iter}]: {metrics_str}")

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
            if "dice" not in metric_name.lower():
                ax.set_yscale("log")
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

    def should_run_validation_partial_epoch(self):
        return self.current_iter % TrainConfig.partial_val_interval == 0

    def should_plot_losses(self):
        return self.current_iter % TrainConfig.plot_losses == 0

    def should_calc_dice(self):
        return self.current_iter % TrainConfig.dice_calc_interval == 0

    def should_print_volume_report(self):
        return self.current_iter % TrainConfig.volume_report_interval == 0

    def get_latest_val_dice(self) -> Optional[float]:
        """Get the most recent validation dice_mean score."""
        dice_vals = self.data["val"]["metrics"].get("dice_mean", [])
        if dice_vals:
            return dice_vals[-1]
        return None


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
    """Linear warmup for encoder (param group 0), constant LR for head (param group 1)."""
    def lr_lambda_encoder(step):
        if warmup_steps == 0:
            return 1.0
        if step < warmup_steps:
            return step / warmup_steps
        return 1.0

    def lr_lambda_head(step):
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, [lr_lambda_encoder, lr_lambda_head]
    )


def train_one_epoch(
    model,
    optimizer,
    scaler,
    train_dataloader,
    val_dataloader,
    metrics: Metrics,
    scheduler=None,
):
    """Training epoch - processes volumes in chunks."""
    model.train()

    for batch_idx, batch in enumerate(train_dataloader):
        imgs = batch['frames']
        labels = batch['label']

        step_metrics, preds = one_training_step(
            imgs,
            labels,
            model,
            optimizer,
            scaler,
            metrics,
        )
        if scheduler is not None:
            scheduler.step()

        metrics.append_regression(preds, labels)
        step_metrics.update(metrics.get_regression_metrics())
        metrics.append("train", step_metrics)

        if metrics.should_plot_losses():
            metrics.plot_metrics()



@typechecked
def one_training_step(
    images: Float[Tensor, "batches channels frames height width"],
    labels: Float[Tensor, "batches"],
    model: nn.Module,
    optimizer: Optimizer,
    scaler: GradScaler,
    metrics: Metrics,
) -> tuple[dict[str, float], Tensor]:
    """Single training step on a volume chunk."""
    optimizer.zero_grad()


    with torch.amp.autocast("cuda"):
        # Get predictions: (1, T, num_classes, H, W)
        logits = model(images.half().cuda())
        alpha = logits[0]
        beta = logits[1]

        loss = beta_nll_loss(alpha, beta, labels.cuda())

        pred = Beta(alpha, beta).mean


    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()

    result = {"loss": np.exp(loss.item())}
    return result, pred


def validation_epoch(model, dataloader, metrics: Metrics, split):
    """Full validation epoch."""
    from collections import defaultdict

    print(f"Running validation: {split}")
    model.eval()

    results = defaultdict(lambda: 0.0)
    num_steps = 0

    with torch.no_grad():
        for batch in dataloader:
            ims = batch["frames"].permute(1, 0, 2, 3).cuda()
            height = batch["label"][0].permute(1, 0, 2).cuda()
            S, C, H, W = ims.shape

            for run in range(0, S, TrainConfig.step_size):
                end_idx = min(run + TrainConfig.step_size, S)
                chunk_size = end_idx - run

                # Pad if necessary
                if chunk_size % TrainConfig.t_patch_size != 0:
                    pad_size = TrainConfig.t_patch_size - (chunk_size % TrainConfig.t_patch_size)
                    chunk_ims = ims[run:end_idx]
                    chunk_height = height[run:end_idx]
                    chunk_ims = torch.cat([chunk_ims, chunk_ims[-1:].expand(pad_size, -1, -1, -1)], dim=0)
                    chunk_height = torch.cat([chunk_height, chunk_height[-1:].expand(pad_size, -1, -1)], dim=0)
                else:
                    chunk_ims = ims[run:end_idx]
                    chunk_height = height[run:end_idx]

                local_metrics = one_validation_step(
                    model,
                    chunk_ims,
                    chunk_height,
                    metrics,
                    num_steps == 0,
                    split,
                )
                for k, v in local_metrics.items():
                    results[k] += v
                num_steps += 1

    for k, v in results.items():
        results[k] = v / max(num_steps, 1)

    metrics.append(split, dict(results))
    metrics.print_latest(split)
    metrics.plot_metrics()


def validation_partial_epoch(model, dataloader, metrics: Metrics, split):
    """Partial validation on subset of data."""
    from collections import defaultdict

    print("Running Partial Validation")
    model.eval()

    results = defaultdict(lambda: 0.0)
    num_steps = 0
    max_vols = 10

    with torch.no_grad():
        for vol_idx, batch in enumerate(dataloader):
            if vol_idx >= max_vols:
                break

            ims = batch["frames"].permute(1, 0, 2, 3).cuda()
            height = batch["label"][0].permute(1, 0, 2).cuda()
            S, C, H, W = ims.shape

            # Just process first chunk for speed
            end_idx = min(TrainConfig.step_size, S)
            chunk_size = end_idx

            if chunk_size % TrainConfig.t_patch_size != 0:
                pad_size = TrainConfig.t_patch_size - (chunk_size % TrainConfig.t_patch_size)
                chunk_ims = ims[:end_idx]
                chunk_height = height[:end_idx]
                chunk_ims = torch.cat([chunk_ims, chunk_ims[-1:].expand(pad_size, -1, -1, -1)], dim=0)
                chunk_height = torch.cat([chunk_height, chunk_height[-1:].expand(pad_size, -1, -1)], dim=0)
            else:
                chunk_ims = ims[:end_idx]
                chunk_height = height[:end_idx]

            local_metrics = one_validation_step(
                model,
                chunk_ims,
                chunk_height,
                metrics,
                vol_idx == 0,
                split,
            )
            for k, v in local_metrics.items():
                results[k] += v
            num_steps += 1

    for k, v in results.items():
        results[k] = v / max(num_steps, 1)

    metrics.append(split, dict(results))
    metrics.print_latest(split)
    metrics.plot_metrics()
    metrics.save("octcube_metrics.json")
    model.train()




def full_supervised_run():
    """Two-phase training: head-only then end-to-end with encoder warmup."""
    os.makedirs("selected_images", exist_ok=True)
    os.makedirs(TrainConfig.save_dir, exist_ok=True)

    train_loader = torch.utils.data.DataLoader(
        HVFDataset(
            split_label="train",
            target_size=(TrainConfig.img_size, TrainConfig.img_size),
            normalize=True
        ),
        batch_size=TrainConfig.batch_size,
        num_workers=TrainConfig.num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        HVFDataset(
            split_label="val",
            target_size=(TrainConfig.img_size, TrainConfig.img_size),
            normalize=True
        ),
        batch_size=TrainConfig.batch_size,
        num_workers=TrainConfig.num_workers,
    )
    print("Loaded data loaders")

    # Create model with encoder FROZEN for phase 1
    model = OCTCubeRegression(
        img_size=TrainConfig.img_size,
        patch_size=TrainConfig.patch_size,
        num_frames=TrainConfig.num_frames,
        t_patch_size=TrainConfig.t_patch_size,
        size=TrainConfig.model_size,
        freeze_encoder=True,
        checkpoint_path=TrainConfig.checkpoint_path,
    ).cuda()

    metrics = Metrics()
    scaler = GradScaler("cuda")

    latest_path = os.path.join(TrainConfig.save_dir, "latest.pt")
    phase1_path = os.path.join(TrainConfig.save_dir, "phase1_final.pt")

    # ----------------------------------------------------------------
    # Phase 1: Train head only (encoder frozen)
    # ----------------------------------------------------------------
    print("=" * 60)
    print(f"Phase 1: Training head only for {TrainConfig.phase1_epochs} epochs")
    print("=" * 60)

    optimizer = torch.optim.AdamW(
        model.head.parameters(),
        lr=TrainConfig.head_lr,
    )

    for e in range(1, TrainConfig.phase1_epochs + 1):
        metrics.current_epoch = e
        print(f"Phase 1 - Epoch {e}/{TrainConfig.phase1_epochs}")
        train_one_epoch(
            model, optimizer, scaler,
            train_loader, val_loader, metrics,
        )
        save_checkpoint(model, optimizer, scaler, metrics, latest_path)

    save_checkpoint(model, optimizer, scaler, metrics, phase1_path)
    print(f"Phase 1 complete. Head checkpoint saved to {phase1_path}")

    # ----------------------------------------------------------------
    # Phase 2: End-to-end fine-tuning (encoder unfrozen + LR warmup)
    # ----------------------------------------------------------------
    print("=" * 60)
    print(f"Phase 2: End-to-end fine-tuning for {TrainConfig.phase2_epochs} epochs")
    print(f"  Encoder LR warmup over first {TrainConfig.warmup_fraction:.0%} of steps")
    print("=" * 60)

    # Unfreeze encoder
    for p in model.encoder.parameters():
        p.requires_grad = True
    print("Encoder unfrozen")

    # New optimizer with separate LRs
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": TrainConfig.encoder_lr},
        {"params": model.head.parameters(), "lr": TrainConfig.head_lr},
    ])
    scaler = GradScaler("cuda")

    # Warmup scheduler for encoder LR
    total_phase2_steps = TrainConfig.phase2_epochs * len(train_loader)
    warmup_steps = int(TrainConfig.warmup_fraction * total_phase2_steps)
    scheduler = get_warmup_scheduler(optimizer, warmup_steps)
    print(f"Warmup: {warmup_steps} steps out of {total_phase2_steps} total")

    for e in range(1, TrainConfig.phase2_epochs + 1):
        epoch_num = TrainConfig.phase1_epochs + e
        metrics.current_epoch = epoch_num
        print(f"Phase 2 - Epoch {e}/{TrainConfig.phase2_epochs} (overall {epoch_num})")
        train_one_epoch(
            model, optimizer, scaler,
            train_loader, val_loader, metrics,
            scheduler=scheduler,
        )
        save_checkpoint(model, optimizer, scaler, metrics, latest_path, save_encoder=True)

    # Save final
    metrics.save(os.path.join(TrainConfig.save_dir, "metrics_final.json"))
    metrics.plot_metrics()
    print("Training complete.")


if __name__ == "__main__":
    full_supervised_run()
