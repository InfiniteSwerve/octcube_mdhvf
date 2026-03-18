"""
End-to-end fine-tuning of OCTCube for HVF MTD regression.

All encoder layers unfrozen with layer-wise learning rate decay (LLRD).
Linear head with dropout, matching the official OCTCubeM fine-tuning recipe.

Launch:  torchrun --nproc_per_node=4 finetune.py
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import time
import warnings
from dataclasses import dataclass
from typing import Optional
from collections import deque, defaultdict
from tqdm import tqdm

from torch import Tensor
from jaxtyping import Float, jaxtyped
from beartype import beartype
from einops import rearrange

from octcube import OCTCubeWrapper
from dataset import HVFDataset

typechecked = jaxtyped(typechecker=beartype)


def _is_distributed():
    return dist.is_available() and dist.is_initialized()


def _rank():
    return dist.get_rank() if _is_distributed() else 0


def _world_size():
    return dist.get_world_size() if _is_distributed() else 1


def _is_main():
    return _rank() == 0


def _setup_distributed():
    """Initialize DDP if launched via torchrun."""
    if "RANK" not in os.environ:
        return  # single-GPU fallback
    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def _cleanup_distributed():
    if _is_distributed():
        dist.destroy_process_group()


# ── Config ────────────────────────────────────────────────────────────

@dataclass
class Config:
    # Model
    img_size: int = 256
    patch_size: int = 16
    num_frames: int = 48
    t_patch_size: int = 3
    model_size: str = "large"
    center_crop_frac: float = 0.5

    # Training
    epochs: int = 20
    batch_size: int = 2
    grad_accum_steps: int = 32
    num_workers: int = 25

    # Optimizer (following official recipe)
    base_lr: float = 1e-3       # Scaled by effective_batch / 256
    weight_decay: float = 0.05
    layer_decay: float = 0.75   # LR multiplier per layer (earlier = smaller LR)
    min_lr: float = 1e-6
    warmup_epochs: int = 5

    # Head
    dropout: float = 0.5

    # Logging / checkpoints
    plot_interval: int = 10       # Plot every N optimizer steps
    val_interval: int = 500       # Partial val every N optimizer steps
    val_max_volumes: int = 100
    save_dir: str = "checkpoints_finetune"
    scatter_dir: str = "scatter_plots_finetune"
    plot_dir: str = "plots_octcube"
    box_bins: int = 10            # Number of bins for box-and-whisker on scatter

    # Paths
    checkpoint_path: Optional[str] = "/storage2/fs1/leeay/Active/jstrand/projects/OCTCubeM/ckpt/OCTCube.pth"


# ── Layer-wise LR decay ──────────────────────────────────────────────

def _get_layer_id(name: str, num_layers: int) -> int:
    """Assign a layer index to each parameter for LLRD.

    Layer 0 = patch_embed / pos_embed / cls_token (earliest, smallest LR).
    Layers 1..num_layers = transformer blocks.
    Layer num_layers+1 = head (highest LR, though head is separate group).
    """
    if name.startswith("patch_embed") or "pos_embed" in name or "cls_token" in name:
        return 0
    elif name.startswith("blocks."):
        block_idx = int(name.split(".")[1])
        return block_idx + 1
    elif name.startswith("norm."):
        return num_layers
    else:
        return num_layers + 1


def build_param_groups(encoder: OCTCubeWrapper, head: nn.Module, cfg: Config):
    """Build optimizer param groups with layer-wise LR decay.

    Follows the BEiT / MAE fine-tuning convention:
    - lr_scale = layer_decay^(num_layers - layer_id)
    - No weight decay on biases, LayerNorm, pos_embed, cls_token.
    """
    eff_batch = cfg.batch_size * cfg.grad_accum_steps * _world_size()
    lr = cfg.base_lr * eff_batch / 256
    if _is_main():
        print(f"Effective LR: {cfg.base_lr} * {eff_batch}/256 = {lr:.6f}")

    num_layers = len(encoder.model.blocks)
    no_decay_keywords = {"bias", "pos_embed", "cls_token"}
    no_decay_types = (nn.LayerNorm,)

    groups = {}  # group_name -> {lr_scale, weight_decay, params}

    for name, param in encoder.model.named_parameters():
        if not param.requires_grad:
            continue

        layer_id = _get_layer_id(name, num_layers)
        lr_scale = cfg.layer_decay ** (num_layers - layer_id)

        # Determine weight decay
        is_no_decay = any(kw in name for kw in no_decay_keywords) or param.ndim == 1
        wd = 0.0 if is_no_decay else cfg.weight_decay

        group_name = f"layer_{layer_id}_wd{wd:.3f}"
        if group_name not in groups:
            groups[group_name] = {
                "lr": lr * lr_scale,
                "weight_decay": wd,
                "params": [],
                "name": group_name,
            }
        groups[group_name]["params"].append(param)

    # Head params — highest LR (no layer decay)
    head_decay = []
    head_no_decay = []
    for name, param in head.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 1 or "bias" in name:
            head_no_decay.append(param)
        else:
            head_decay.append(param)

    if head_decay:
        groups["head_decay"] = {"lr": lr, "weight_decay": cfg.weight_decay, "params": head_decay}
    if head_no_decay:
        groups["head_no_decay"] = {"lr": lr, "weight_decay": 0.0, "params": head_no_decay}

    param_groups = list(groups.values())

    # Print summary
    total_params = sum(p.numel() for g in param_groups for p in g["params"])
    if _is_main():
        print(f"Optimizer: {len(param_groups)} param groups, {total_params:,} trainable params")
        for g in sorted(param_groups, key=lambda g: g["lr"]):
            n = sum(p.numel() for p in g["params"])
            print(f"  {g.get('name', 'head'):30s}  lr={g['lr']:.2e}  wd={g['weight_decay']:.3f}  params={n:>10,}")

    return param_groups, lr


# ── Cosine schedule with warmup ──────────────────────────────────────

def build_scheduler(optimizer, cfg: Config, steps_per_epoch: int):
    """Cosine annealing with linear warmup, per-iteration stepping."""
    total_steps = cfg.epochs * steps_per_epoch
    warmup_steps = cfg.warmup_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(cfg.min_lr / cfg.base_lr, 0.5 * (1 + np.cos(np.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── Model ─────────────────────────────────────────────────────────────

class OCTCubeFinetune(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.encoder = OCTCubeWrapper(
            img_size=cfg.img_size,
            patch_size=cfg.patch_size,
            in_chans=1,
            num_frames=cfg.num_frames,
            t_patch_size=cfg.t_patch_size,
            size=cfg.model_size,
        )
        if cfg.checkpoint_path:
            self.encoder.load_pretrained(cfg.checkpoint_path)

        self.dropout = nn.Dropout(cfg.dropout)
        self.head = nn.Linear(self.encoder.embed_dim, 1)
        nn.init.trunc_normal_(self.head.weight, std=2e-5)
        nn.init.zeros_(self.head.bias)

    @beartype
    def forward(
        self, x: Float[Tensor, "B C T H W"] | Float[Tensor, "B T H W"],
    ) -> Float[Tensor, "B"]:
        if x.dim() == 4:
            x = x.unsqueeze(1)
        # x is (B, C, T, H, W); encoder expects (B, T, C, H, W)
        x = rearrange(x, "b c t h w -> b t c h w")

        # Global average pool over all patch tokens (no cls token)
        features = self.encoder(x, return_all_tokens=False)  # (B, D)
        features = self.dropout(features)
        return self.head(features).squeeze(-1)  # (B,)


# ── Metrics ───────────────────────────────────────────────────────────

class Metrics:
    # Hodapp-Parrish-Anderson severity thresholds (MTD in dB)
    HODAPP_THRESHOLDS = [0, -6, -12]
    HODAPP_LABELS = ["Normal", "Early", "Moderate", "Advanced"]

    def __init__(self, plot_dir="plots_octcube", box_bins=10, unscale_fn=None):
        self.plot_dir = plot_dir
        self.box_bins = box_bins
        self.unscale_fn = unscale_fn  # callable: normalized -> original dB scale
        os.makedirs(plot_dir, exist_ok=True)
        self.data = {}
        for split in ["train", "val"]:
            self.data[split] = {"iterations": [], "metrics": defaultdict(list)}
        self.opt_step = 0
        self.epoch = 0
        self.rolling_preds = deque(maxlen=2000)
        self.rolling_gts = deque(maxlen=2000)
        self.rolling_val_preds = deque(maxlen=2000)
        self.rolling_val_gts = deque(maxlen=2000)
        self.eta_str = ""
        self.best_val_r2 = -float("inf")
        self.best_val_epoch = -1
        self._full_val_preds = None
        self._full_val_gts = None
        self._best_val_preds = None
        self._best_val_gts = None

    def append(self, split, metrics_dict):
        self.data[split]["iterations"].append(self.opt_step)
        for k, v in metrics_dict.items():
            self.data[split]["metrics"][k].append(v)

    def append_regression(self, preds, labels):
        p = preds.detach().cpu().flatten().tolist()
        g = labels.detach().cpu().flatten().tolist()
        self.rolling_preds.extend(p)
        self.rolling_gts.extend(g)

    def get_regression_metrics(self):
        if len(self.rolling_preds) < 10:
            return {"mae": 0, "pearson_r": 0, "r2": 0}
        p = np.array(self.rolling_preds)
        g = np.array(self.rolling_gts)
        mae = np.abs(p - g).mean()
        r = np.corrcoef(p, g)[0, 1] if np.std(p) > 1e-8 else 0.0
        ss_res = ((p - g) ** 2).sum()
        ss_tot = ((g - g.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return {"mae": float(mae), "pearson_r": float(r), "r2": float(r2)}

    def plot(self):
        keys = ["loss", "mae", "pearson_r", "r2", "lr", "pred_std"]
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        for split, color in [("train", "C0"), ("val", "C1")]:
            iters = self.data[split]["iterations"]
            for ax, key in zip(axes.flat, keys):
                vals = self.data[split]["metrics"].get(key, [])
                if vals and len(vals) == len(iters):
                    ax.plot(iters, vals, color=color, alpha=0.6, label=split, linewidth=0.8)
                    ax.set_title(key)
                    ax.legend(fontsize=7)
        fig.suptitle(f"Epoch {self.epoch} | step {self.opt_step} | {self.eta_str}", fontsize=10)
        fig.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, "finetune_metrics_octcube.png"), dpi=120)
        plt.close()

    def append_val_regression(self, preds, gts):
        """Append to rolling validation scatter from periodic val checks."""
        if isinstance(preds, np.ndarray):
            preds = preds.tolist()
            gts = gts.tolist()
        self.rolling_val_preds.extend(preds)
        self.rolling_val_gts.extend(gts)

    def set_full_val(self, preds, gts):
        """Store full end-of-epoch validation results."""
        self._full_val_preds = preds
        self._full_val_gts = gts

    def update_best_val(self, preds, gts):
        """Store best validation preds/gts and save dedicated best-val plots."""
        self._best_val_preds = preds
        self._best_val_gts = gts
        self.best_val_epoch = self.epoch
        self._save_best_val_plots()

    def _save_best_val_plots(self):
        """Save scatter+box, Hodapp scatter, and confusion matrix for best val epoch."""
        if self._best_val_preds is None:
            return
        gt = self._best_val_gts
        pred = self._best_val_preds
        tag = f"Best Val (n={len(pred)}, epoch {self.epoch})"
        # 1) Box scatter (original dB scale)
        gt_db, pred_db = self._to_db(gt, pred)
        fig, ax = plt.subplots(1, 1, figsize=(7, 6))
        self._scatter_panel(ax, gt_db, pred_db, tag, box_bins=self.box_bins)
        fig.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, "finetune_scatter_octcube_best.png"), dpi=150)
        plt.close()
        # 2) Hodapp scatter
        fig, ax = plt.subplots(1, 1, figsize=(7, 6))
        self._hodapp_scatter_panel(ax, gt_db, pred_db, tag)
        fig.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, "finetune_hodapp_octcube_best.png"), dpi=150)
        plt.close()
        # 3) Confusion matrix
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        self._hodapp_confusion_panel(ax, gt_db, pred_db, tag)
        fig.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, "finetune_confusion_octcube_best.png"), dpi=150)
        plt.close()

    def plot_scatter(self):
        """Plot up to 3 panels: rolling train, rolling val, full epoch val.
        Produces three files: box scatter, Hodapp scatter, confusion matrix."""
        panels = []
        if len(self.rolling_preds) >= 20:
            panels.append((np.array(self.rolling_gts), np.array(self.rolling_preds),
                           f"Train rolling (n={len(self.rolling_preds)})"))
        if len(self.rolling_val_preds) >= 10:
            panels.append((np.array(self.rolling_val_gts), np.array(self.rolling_val_preds),
                           f"Val rolling (n={len(self.rolling_val_preds)})"))
        if self._full_val_preds is not None:
            panels.append((self._full_val_gts, self._full_val_preds,
                           f"Val full epoch (n={len(self._full_val_preds)})"))
        if not panels:
            return
        # Convert all panels to dB scale
        panels_db = [(self._to_db(gt, pred) + (title,)) for gt, pred, title in panels]
        # 1) Box scatter
        fig, axes = plt.subplots(1, len(panels_db), figsize=(5.5 * len(panels_db), 5))
        if len(panels_db) == 1:
            axes = [axes]
        for ax, (gt, pred, title) in zip(axes, panels_db):
            self._scatter_panel(ax, gt, pred, title, box_bins=self.box_bins)
        fig.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, "finetune_scatter_octcube.png"), dpi=120)
        plt.close()
        # 2) Hodapp scatter
        fig, axes = plt.subplots(1, len(panels_db), figsize=(5.5 * len(panels_db), 5))
        if len(panels_db) == 1:
            axes = [axes]
        for ax, (gt, pred, title) in zip(axes, panels_db):
            self._hodapp_scatter_panel(ax, gt, pred, title)
        fig.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, "finetune_hodapp_octcube.png"), dpi=120)
        plt.close()
        # 3) Confusion matrix (use last panel with most data)
        gt_db, pred_db, title = panels_db[-1]
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        self._hodapp_confusion_panel(ax, gt_db, pred_db, title)
        fig.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, "finetune_confusion_octcube.png"), dpi=120)
        plt.close()

    @staticmethod
    def _scatter_panel(ax, gt, pred, title, box_bins=10):
        mae = np.abs(pred - gt).mean()
        r = np.corrcoef(pred, gt)[0, 1] if np.std(pred) > 1e-8 else 0.0
        ss_res = ((pred - gt) ** 2).sum()
        ss_tot = ((gt - gt.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        ax.scatter(gt, pred, s=4, alpha=0.4, zorder=2)
        lo, hi = min(gt.min(), pred.min()), max(gt.max(), pred.max())
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1, zorder=3)
        # Box-and-whisker overlay
        if len(gt) >= box_bins:
            edges = np.linspace(gt.min(), gt.max(), box_bins + 1)
            centers = []
            groups = []
            for i in range(box_bins):
                mask = (gt >= edges[i]) & (gt < edges[i + 1])
                if i == box_bins - 1:
                    mask = (gt >= edges[i]) & (gt <= edges[i + 1])
                if mask.sum() >= 2:
                    centers.append((edges[i] + edges[i + 1]) / 2)
                    groups.append(pred[mask])
            if groups:
                width = (edges[1] - edges[0]) * 0.6
                bp = ax.boxplot(groups, positions=centers, widths=width,
                                patch_artist=True, manage_ticks=False, zorder=4)
                for box in bp["boxes"]:
                    box.set(facecolor="C1", alpha=0.3)
                for median in bp["medians"]:
                    median.set(color="C3", linewidth=1.5)
                for whisker in bp["whiskers"]:
                    whisker.set(color="C1", alpha=0.5)
                for cap in bp["caps"]:
                    cap.set(color="C1", alpha=0.5)
                for flier in bp["fliers"]:
                    flier.set(marker=".", markersize=2, alpha=0.3)
        ax.set_xlabel("GT (dB)")
        ax.set_ylabel("Pred (dB)")
        ax.set_title(f"{title}\nMAE={mae:.4f}  r={r:.4f}  R²={r2:.4f}")

    def _to_db(self, gt, pred):
        """Convert normalized arrays back to original dB scale."""
        if self.unscale_fn is not None:
            return self.unscale_fn(gt), self.unscale_fn(pred)
        return gt, pred

    @staticmethod
    def _hodapp_class(vals):
        """Classify MTD values (dB) into Hodapp severity: 0=Normal, 1=Early, 2=Moderate, 3=Advanced."""
        cls = np.full(len(vals), 3, dtype=int)  # default Advanced
        cls[vals > -12] = 2  # Moderate
        cls[vals > -6] = 1   # Early
        cls[vals > 0] = 0    # Normal
        return cls

    @staticmethod
    def _hodapp_scatter_panel(ax, gt, pred, title):
        """Scatter with Hodapp-Parrish-Anderson criterion lines (no box plot)."""
        mae = np.abs(pred - gt).mean()
        r = np.corrcoef(pred, gt)[0, 1] if np.std(pred) > 1e-8 else 0.0
        ss_res = ((pred - gt) ** 2).sum()
        ss_tot = ((gt - gt.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        ax.scatter(gt, pred, s=4, alpha=0.4, zorder=2)
        lo, hi = min(gt.min(), pred.min()), max(gt.max(), pred.max())
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1, zorder=3)
        # Hodapp criterion lines
        for thresh in [0, -6, -12]:
            if lo <= thresh <= hi:
                ax.axhline(thresh, color="gray", linestyle=":", linewidth=0.8, alpha=0.7, zorder=1)
                ax.axvline(thresh, color="gray", linestyle=":", linewidth=0.8, alpha=0.7, zorder=1)
        # Label the regions along GT axis
        for thresh, lbl in [(0, "Normal"), (-6, "Early"), (-12, "Moderate")]:
            if lo <= thresh <= hi:
                ax.text(thresh, hi - (hi - lo) * 0.02, f"  {lbl}", fontsize=7,
                        color="gray", ha="left", va="top", zorder=5)
        if lo < -12:
            ax.text(lo + (hi - lo) * 0.01, hi - (hi - lo) * 0.02, "Adv.",
                    fontsize=7, color="gray", ha="left", va="top", zorder=5)
        ax.set_xlabel("GT (dB)")
        ax.set_ylabel("Pred (dB)")
        ax.set_title(f"{title}\nMAE={mae:.4f}  r={r:.4f}  R²={r2:.4f}")

    @classmethod
    def _hodapp_confusion_panel(cls, ax, gt, pred, title):
        """4x4 confusion matrix based on Hodapp severity classification."""
        gt_cls = cls._hodapp_class(gt)
        pred_cls = cls._hodapp_class(pred)
        n_classes = len(cls.HODAPP_LABELS)
        cm = np.zeros((n_classes, n_classes), dtype=int)
        for g, p in zip(gt_cls, pred_cls):
            cm[g, p] += 1
        # Normalize rows for color intensity (recall per class)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.where(row_sums > 0, cm / row_sums, 0)
        ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="equal")
        # Annotate cells with count and percentage
        for i in range(n_classes):
            for j in range(n_classes):
                pct = cm_norm[i, j] * 100
                color = "white" if cm_norm[i, j] > 0.5 else "black"
                ax.text(j, i, f"{cm[i, j]}\n{pct:.0f}%", ha="center", va="center",
                        fontsize=9, color=color)
        ax.set_xticks(range(n_classes))
        ax.set_yticks(range(n_classes))
        ax.set_xticklabels(cls.HODAPP_LABELS, fontsize=8)
        ax.set_yticklabels(cls.HODAPP_LABELS, fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        accuracy = (gt_cls == pred_cls).mean()
        ax.set_title(f"{title}\nHodapp Confusion (acc={accuracy:.1%})")

    def save_epoch_scatter(self, val_preds, val_gts):
        """Save scatter, Hodapp scatter, and confusion matrix for this epoch."""
        gt_db, pred_db = self._to_db(val_gts, val_preds)
        tag = f"Val epoch {self.epoch} (n={len(val_preds)})"
        # 1) Box scatter
        fig, ax = plt.subplots(1, 1, figsize=(7, 6))
        self._scatter_panel(ax, gt_db, pred_db, tag, box_bins=self.box_bins)
        fig.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, f"finetune_scatter_octcube_epoch{self.epoch}.png"), dpi=150)
        plt.close()
        # 2) Hodapp scatter
        fig, ax = plt.subplots(1, 1, figsize=(7, 6))
        self._hodapp_scatter_panel(ax, gt_db, pred_db, tag)
        fig.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, f"finetune_hodapp_octcube_epoch{self.epoch}.png"), dpi=150)
        plt.close()
        # 3) Confusion matrix
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        self._hodapp_confusion_panel(ax, gt_db, pred_db, tag)
        fig.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, f"finetune_confusion_octcube_epoch{self.epoch}.png"), dpi=150)
        plt.close()

    def save_epoch_metrics(self, val_metrics, path_dir):
        """Save metrics JSON for this epoch."""
        import json
        path = os.path.join(path_dir, f"metrics_epoch{self.epoch}.json")
        with open(path, "w") as f:
            json.dump({
                "opt_step": self.opt_step,
                "epoch": self.epoch,
                "val_metrics": val_metrics,
                "data": {s: {"iterations": d["iterations"], "metrics": dict(d["metrics"])}
                         for s, d in self.data.items()},
            }, f)

    def refresh_best_val_with_full_data(self, val_preds, val_gts):
        """Re-save best val plots with full validation data if best was this epoch."""
        if self.best_val_epoch == self.epoch:
            self._best_val_preds = val_preds
            self._best_val_gts = val_gts
            self._save_best_val_plots()

    def save(self, path):
        import json
        with open(path, "w") as f:
            json.dump({
                "opt_step": self.opt_step,
                "epoch": self.epoch,
                "data": {s: {"iterations": d["iterations"], "metrics": dict(d["metrics"])}
                         for s, d in self.data.items()},
            }, f)


# ── Training ──────────────────────────────────────────────────────────

def _format_time(s):
    s = int(s)
    return f"{s//3600}h{(s%3600)//60:02d}m" if s >= 3600 else f"{s//60}m{s%60:02d}s"


@typechecked
def forward_step(
    images: Float[Tensor, "B C T H W"],
    labels: Float[Tensor, "B"],
    model: nn.Module,
    accum: int,
) -> tuple[dict[str, float], Tensor]:
    device = next(model.parameters()).device
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        pred = model(images.to(device)).float()
        loss = F.mse_loss(pred, labels.to(device).float())
    (loss / accum).backward()
    return {"loss": loss.item(), "pred_std": pred.detach().std().item()}, pred.detach().cpu()


def validate(model, loader, cfg: Config, max_volumes: int | None = None):
    eval_model = model.module if isinstance(model, DDP) else model
    eval_model.eval()
    device = next(eval_model.parameters()).device
    all_preds, all_labels = [], []
    total_loss, n = 0.0, 0
    total_batches = len(loader) if max_volumes is None else min(len(loader), max_volumes)
    pbar = tqdm(loader, total=total_batches, desc="Validating",
                disable=not _is_main(), leave=False)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for batch in pbar:
            if max_volumes is not None and n >= max_volumes:
                break
            imgs = batch["frames"].to(device)
            labels = batch["label"].to(device)
            pred = eval_model(imgs).float()
            total_loss += F.mse_loss(pred, labels.float()).item()
            all_preds.append(pred.cpu())
            all_labels.append(batch["label"])
            n += imgs.shape[0]
            if _is_main():
                pbar.set_postfix_str(f"loss={total_loss/n:.4f} n={n}", refresh=False)
    preds = torch.cat(all_preds)
    gts = torch.cat(all_labels)
    mae = (preds - gts).abs().mean().item()
    r = torch.corrcoef(torch.stack([preds, gts]))[0, 1].item()
    ss_res = ((preds - gts) ** 2).sum().item()
    ss_tot = ((gts - gts.mean()) ** 2).sum().item()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {
        "loss": total_loss / max(n, 1),
        "mae": mae,
        "pearson_r": r,
        "r2": r2,
        "pred_std": preds.std().item(),
    }, preds.numpy(), gts.numpy()


def save_checkpoint(model, optimizer, scheduler, metrics, path,
                    val_preds=None, val_gts=None):
    if not _is_main():
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    raw_model = model.module if isinstance(model, DDP) else model
    ckpt = {
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "opt_step": metrics.opt_step,
        "epoch": metrics.epoch,
    }
    if val_preds is not None:
        ckpt["val_preds"] = val_preds
        ckpt["val_gts"] = val_gts
    torch.save(ckpt, path)
    print(f"Saved checkpoint to {path}")


def train():
    _setup_distributed()
    rank = _rank()
    world = _world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")

    cfg = Config()
    # Scale grad_accum down by world_size to keep same effective batch size
    # but finish faster.  effective_batch = batch_size * grad_accum * world_size
    cfg.grad_accum_steps = max(1, cfg.grad_accum_steps // world)
    # Scale workers down so total across ranks doesn't exceed CPU count
    cfg.num_workers = max(1, cfg.num_workers // world)
    accum = cfg.grad_accum_steps

    if _is_main():
        os.makedirs(cfg.save_dir, exist_ok=True)
        os.makedirs(cfg.scatter_dir, exist_ok=True)
        os.makedirs(cfg.plot_dir, exist_ok=True)

    # Data
    ds_kwargs = dict(
        target_size=(cfg.img_size, cfg.img_size),
        normalize=True,
        center_crop_frac=cfg.center_crop_frac,
    )
    train_ds = HVFDataset(split_label="train", **ds_kwargs)
    val_ds = HVFDataset(split_label="val", **ds_kwargs)

    train_sampler = DistributedSampler(train_ds, num_replicas=world, rank=rank,
                                       shuffle=True) if world > 1 else None
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    # Model
    model = OCTCubeFinetune(cfg).to(device)

    # Optimizer with LLRD (build before DDP wrap so param names don't have "module." prefix)
    param_groups, lr = build_param_groups(model.encoder, model.head, cfg)
    optimizer = torch.optim.AdamW(param_groups)

    # Wrap in DDP
    if world > 1:
        model = DDP(model, device_ids=[local_rank])

    # Scheduler
    opt_steps_per_epoch = len(train_loader) // accum
    scheduler = build_scheduler(optimizer, cfg, opt_steps_per_epoch)

    metrics = Metrics(plot_dir=cfg.plot_dir, box_bins=cfg.box_bins,
                      unscale_fn=train_ds.unscale_label)

    if _is_main():
        eff_batch = cfg.batch_size * accum * world
        print("=" * 60)
        print(f"Fine-tuning OCTCube-{cfg.model_size} end-to-end  ({world} GPU{'s' if world > 1 else ''})")
        print(f"  img_size={cfg.img_size}, batch/gpu={cfg.batch_size}, accum={accum}, world={world}")
        print(f"  effective_batch={eff_batch}")
        print(f"  epochs={cfg.epochs}, warmup={cfg.warmup_epochs}")
        print(f"  base_lr={cfg.base_lr}, eff_lr={lr:.6f}, layer_decay={cfg.layer_decay}")
        print(f"  weight_decay={cfg.weight_decay}, dropout={cfg.dropout}")
        print(f"  {len(train_loader)} batches/epoch, {opt_steps_per_epoch} opt steps/epoch")
        print("=" * 60)

    for epoch in range(1, cfg.epochs + 1):
        metrics.epoch = epoch
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad()

        accum_loss = 0.0
        accum_preds, accum_labels = [], []
        accum_count = 0
        epoch_start = time.time()
        total_batches = len(train_loader)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs}",
                    leave=True, disable=not _is_main())
        for batch_idx, batch in enumerate(pbar):
            step_metrics, preds = forward_step(
                batch["frames"], batch["label"], model, accum,
            )
            accum_loss += step_metrics["loss"]
            accum_preds.append(preds)
            accum_labels.append(batch["label"])
            accum_count += 1

            if _is_main():
                pbar.set_postfix_str(
                    f"micro={accum_count}/{accum} loss={step_metrics['loss']:.4f}", refresh=False
                )

            is_boundary = (batch_idx + 1) % accum == 0 or (batch_idx + 1) == total_batches
            if is_boundary:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                metrics.opt_step += 1

                if _is_main():
                    all_p = torch.cat(accum_preds)
                    all_l = torch.cat(accum_labels)
                    metrics.append_regression(all_p, all_l)
                    reg = metrics.get_regression_metrics()

                    avg_loss = accum_loss / accum_count
                    log = {
                        "loss": avg_loss,
                        "lr": optimizer.param_groups[-1]["lr"],
                        **reg,
                    }
                    metrics.append("train", log)

                    pbar.set_postfix_str(
                        f"loss={avg_loss:.4f} mae={reg['mae']:.4f} r={reg['pearson_r']:.3f}",
                        refresh=True,
                    )

                    # ETA
                    elapsed = time.time() - epoch_start
                    done = batch_idx + 1
                    remaining = elapsed / done * (total_batches - done)
                    remaining += elapsed / done * total_batches * (cfg.epochs - epoch)
                    metrics.eta_str = f"batch {done}/{total_batches} | {_format_time(elapsed)} elapsed, ETA {_format_time(remaining)}"

                accum_loss = 0.0
                accum_preds, accum_labels = [], []
                accum_count = 0

                if _is_main() and metrics.opt_step % cfg.plot_interval == 0:
                    metrics.plot()
                    metrics.plot_scatter()

                if metrics.opt_step % cfg.val_interval == 0:
                    # All ranks run validation to avoid NCCL timeout
                    val_metrics, vp, vg = validate(model, val_loader, cfg, max_volumes=cfg.val_max_volumes)
                    if _is_main():
                        metrics.append("val", val_metrics)
                        metrics.append_val_regression(vp, vg)
                        metrics.plot()
                        metrics.plot_scatter()
                        if val_metrics["r2"] > metrics.best_val_r2:
                            metrics.best_val_r2 = val_metrics["r2"]
                            metrics.update_best_val(vp, vg)
                            save_checkpoint(model, optimizer, scheduler, metrics,
                                            os.path.join(cfg.save_dir, "best.pt"),
                                            val_preds=vp, val_gts=vg)
                            print(f"  New best val R²={val_metrics['r2']:.4f}")
                    if _is_distributed():
                        dist.barrier()
                    model.train()

        # End-of-epoch full validation (all ranks run to avoid NCCL timeout)
        val_metrics, val_preds, val_gts = validate(model, val_loader, cfg)
        if _is_main():
            metrics.append("val", val_metrics)
            print(f"Epoch {epoch}/{cfg.epochs}  val_loss={val_metrics['loss']:.5f}  "
                  f"mae={val_metrics['mae']:.4f}  r={val_metrics['pearson_r']:.4f}  "
                  f"r2={val_metrics['r2']:.4f}")

            metrics.set_full_val(val_preds, val_gts)
            metrics.plot()
            metrics.plot_scatter()
            metrics.save(os.path.join(cfg.save_dir, "metrics.json"))
            # Per-epoch scatter and metrics
            metrics.save_epoch_scatter(val_preds, val_gts)
            metrics.save_epoch_metrics(val_metrics, cfg.save_dir)
            save_checkpoint(model, optimizer, scheduler, metrics,
                            os.path.join(cfg.save_dir, "latest.pt"),
                            val_preds=val_preds, val_gts=val_gts)
            if val_metrics["r2"] > metrics.best_val_r2:
                metrics.best_val_r2 = val_metrics["r2"]
                metrics.update_best_val(val_preds, val_gts)
                save_checkpoint(model, optimizer, scheduler, metrics,
                                os.path.join(cfg.save_dir, "best.pt"),
                                val_preds=val_preds, val_gts=val_gts)
                print(f"  New best val R²={val_metrics['r2']:.4f}")
            # If best was found mid-epoch with partial data, re-save with full data
            metrics.refresh_best_val_with_full_data(val_preds, val_gts)

        if _is_distributed():
            dist.barrier()
        model.train()

    if _is_main():
        print("Training complete.")
    _cleanup_distributed()


def validate_from_checkpoint(ckpt_path=None, run_inference=True, box_bins=10):
    """Load a checkpoint and regenerate best-val scatter+box plots.

    If the checkpoint already contains saved val_preds/val_gts, plots are
    generated instantly without running inference.  Pass run_inference=True
    to re-run validation on the dataset (slower but always up-to-date).
    """
    cfg = Config()
    cfg.box_bins = box_bins
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if ckpt_path is None:
        ckpt_path = os.path.join(cfg.save_dir, "best.pt")
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Need a dataset instance for unscale_fn
    ds_kwargs = dict(
        target_size=(cfg.img_size, cfg.img_size),
        normalize=True,
        center_crop_frac=cfg.center_crop_frac,
    )
    val_ds = HVFDataset(split_label="val", **ds_kwargs)

    metrics = Metrics(plot_dir=cfg.plot_dir, box_bins=cfg.box_bins,
                      unscale_fn=val_ds.unscale_label)
    metrics.epoch = ckpt.get("epoch", 0)
    metrics.opt_step = ckpt.get("opt_step", 0)

    val_preds = ckpt.get("val_preds")
    val_gts = ckpt.get("val_gts")

    if val_preds is not None and not run_inference:
        print(f"Using cached val preds from checkpoint (n={len(val_preds)})")
    else:
        print("Running validation inference...")
        model = OCTCubeFinetune(cfg).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=cfg.batch_size,
            num_workers=cfg.num_workers, pin_memory=True,
        )
        _, val_preds, val_gts = validate(model, val_loader, cfg)

    # Generate plots
    metrics.update_best_val(val_preds, val_gts)
    metrics.set_full_val(val_preds, val_gts)
    metrics.plot_scatter()
    print(f"Plots saved to {cfg.plot_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true",
                        help="Run validation from checkpoint and generate plots (no training)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint (default: checkpoints_finetune/best.pt)")
    parser.add_argument("--run-inference", action="store_true",
                        help="Force re-running inference instead of using cached preds")
    parser.add_argument("--box-bins", type=int, default=10,
                        help="Number of bins for box-and-whisker plot (default: 10)")
    args = parser.parse_args()

    if args.validate:
        validate_from_checkpoint(
            ckpt_path=args.checkpoint,
            run_inference=args.run_inference,
            box_bins=args.box_bins,
        )
    else:
        train()
