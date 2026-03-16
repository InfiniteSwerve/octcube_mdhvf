"""
EfficientNet3D-B0 baseline for HVF MTD regression from 3D OCT volumes.

Control experiment against OCTCube, following Koyama et al. (Sci Rep 2025):
  - EfficientNet3D-B0, trained from scratch (no pretraining)
  - Input: 224 x 224, 128 frames, grayscale, min-max normalized to [-1, 1]
  - 30% dropout, Adam optimizer, MSE loss
  - Batch size 4 (effective), cosine LR schedule with warmup

Launch:  torchrun --nproc_per_node=4 finetune_efficientnet.py
"""

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
from dataclasses import dataclass
from collections import deque, defaultdict
from tqdm import tqdm

from efficientnet3d import EfficientNet3D
from dataset import HVFDataset


# ── Distributed helpers (same as finetune.py) ────────────────────────

def _is_distributed():
    return dist.is_available() and dist.is_initialized()

def _rank():
    return dist.get_rank() if _is_distributed() else 0

def _world_size():
    return dist.get_world_size() if _is_distributed() else 1

def _is_main():
    return _rank() == 0

def _setup_distributed():
    if "RANK" not in os.environ:
        return
    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def _cleanup_distributed():
    if _is_distributed():
        dist.destroy_process_group()


# ── Config ────────────────────────────────────────────────────────────

@dataclass
class Config:
    # Model (Koyama et al. uses 224x224x128)
    img_size: int = 224
    num_frames: int = 128
    center_crop_frac: float = 0.5
    dropout: float = 0.3         # Paper: 30% dropout

    # Training
    epochs: int = 20
    batch_size: int = 1          # Per-GPU; effective = batch_size * accum * world
    grad_accum_steps: int = 4    # Target effective batch = 4 (paper)

    # Optimizer (Adam, paper uses "variable learning rate schedule")
    lr: float = 1e-3
    weight_decay: float = 1e-4
    min_lr: float = 1e-6
    warmup_epochs: int = 2

    # Workers
    num_workers: int = 25

    # Logging / checkpoints
    plot_interval: int = 10
    val_interval: int = 2000
    val_max_volumes: int = 50
    save_dir: str = "checkpoints_efficientnet"
    scatter_dir: str = "scatter_plots_efficientnet"


# ── Scheduler ─────────────────────────────────────────────────────────

def build_scheduler(optimizer, cfg: Config, steps_per_epoch: int):
    total_steps = cfg.epochs * steps_per_epoch
    warmup_steps = cfg.warmup_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(cfg.min_lr / cfg.lr, 0.5 * (1 + np.cos(np.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── Metrics (reused from finetune.py) ─────────────────────────────────

class Metrics:
    def __init__(self):
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
        self._full_val_preds = None
        self._full_val_gts = None

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
        fig.suptitle(f"EfficientNet3D | Epoch {self.epoch} | step {self.opt_step} | {self.eta_str}", fontsize=10)
        fig.tight_layout()
        plt.savefig("efficientnet_metrics.png", dpi=120)
        plt.close()

    def append_val_regression(self, preds, gts):
        if isinstance(preds, np.ndarray):
            preds = preds.tolist()
            gts = gts.tolist()
        self.rolling_val_preds.extend(preds)
        self.rolling_val_gts.extend(gts)

    def set_full_val(self, preds, gts):
        self._full_val_preds = preds
        self._full_val_gts = gts

    def plot_scatter(self):
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
        fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 5))
        if len(panels) == 1:
            axes = [axes]
        for ax, (gt, pred, title) in zip(axes, panels):
            self._scatter_panel(ax, gt, pred, title)
        fig.tight_layout()
        plt.savefig("efficientnet_scatter.png", dpi=120)
        plt.close()

    @staticmethod
    def _scatter_panel(ax, gt, pred, title):
        mae = np.abs(pred - gt).mean()
        r = np.corrcoef(pred, gt)[0, 1] if np.std(pred) > 1e-8 else 0.0
        ss_res = ((pred - gt) ** 2).sum()
        ss_tot = ((gt - gt.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        ax.scatter(gt, pred, s=4, alpha=0.4)
        lo, hi = min(gt.min(), pred.min()), max(gt.max(), pred.max())
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1)
        ax.set_xlabel("GT")
        ax.set_ylabel("Pred")
        ax.set_title(f"{title}\nMAE={mae:.4f}  r={r:.4f}  R²={r2:.4f}")

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


def forward_step(images, labels, model, accum):
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


def save_checkpoint(model, optimizer, scheduler, metrics, path):
    if not _is_main():
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    raw_model = model.module if isinstance(model, DDP) else model
    torch.save({
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "opt_step": metrics.opt_step,
        "epoch": metrics.epoch,
    }, path)
    print(f"Saved checkpoint to {path}")


def train():
    _setup_distributed()
    rank = _rank()
    world = _world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")

    cfg = Config()
    cfg.grad_accum_steps = max(1, cfg.grad_accum_steps // world)
    cfg.num_workers = max(1, cfg.num_workers // world)
    accum = cfg.grad_accum_steps

    if _is_main():
        os.makedirs(cfg.save_dir, exist_ok=True)
        os.makedirs(cfg.scatter_dir, exist_ok=True)

    # Data — note: different img_size and num_frames from OCTCube
    ds_kwargs = dict(
        target_size=(cfg.img_size, cfg.img_size),
        normalize=True,
        center_crop_frac=cfg.center_crop_frac,
        num_frames=cfg.num_frames,
    )
    train_ds = HVFDataset(split_label="train", **ds_kwargs)
    val_ds = HVFDataset(split_label="val", **ds_kwargs)

    train_sampler = DistributedSampler(train_ds, num_replicas=world, rank=rank,
                                       shuffle=True) if world > 1 else None
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=(train_sampler is None),
        sampler=train_sampler, num_workers=cfg.num_workers, pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers,
        pin_memory=True,
    )

    # Model — from scratch, no pretraining
    model = EfficientNet3D(
        in_channels=1,
        num_classes=1,
        dropout_rate=cfg.dropout,
        first_stride=2,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())

    # Optimizer — Adam (paper), no LLRD needed (training from scratch)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    # Wrap in DDP
    if world > 1:
        model = DDP(model, device_ids=[local_rank])

    # Scheduler
    opt_steps_per_epoch = len(train_loader) // accum
    scheduler = build_scheduler(optimizer, cfg, opt_steps_per_epoch)

    metrics = Metrics()

    if _is_main():
        eff_batch = cfg.batch_size * accum * world
        print("=" * 60)
        print(f"EfficientNet3D-B0 baseline  ({world} GPU{'s' if world > 1 else ''})")
        print(f"  {total_params:,} parameters (training from scratch)")
        print(f"  img_size={cfg.img_size}, num_frames={cfg.num_frames}")
        print(f"  batch/gpu={cfg.batch_size}, accum={accum}, world={world}")
        print(f"  effective_batch={eff_batch}")
        print(f"  epochs={cfg.epochs}, warmup={cfg.warmup_epochs}")
        print(f"  lr={cfg.lr}, weight_decay={cfg.weight_decay}, dropout={cfg.dropout}")
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

            # (micro-batch postfix removed to prevent tqdm flickering)

            is_boundary = (batch_idx + 1) % accum == 0 or (batch_idx + 1) == total_batches
            if is_boundary:
                # Gradient clipping (standard practice)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
                        "lr": optimizer.param_groups[0]["lr"],
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
                    val_metrics, vp, vg = validate(model, val_loader, cfg, max_volumes=cfg.val_max_volumes)
                    if _is_main():
                        metrics.append("val", val_metrics)
                        metrics.append_val_regression(vp, vg)
                        metrics.plot()
                        metrics.plot_scatter()
                        if val_metrics["r2"] > metrics.best_val_r2:
                            metrics.best_val_r2 = val_metrics["r2"]
                            save_checkpoint(model, optimizer, scheduler, metrics,
                                            os.path.join(cfg.save_dir, "best.pt"))
                            print(f"  New best val R²={val_metrics['r2']:.4f}")
                    if _is_distributed():
                        dist.barrier()
                    model.train()

        # End-of-epoch full validation
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
            save_checkpoint(model, optimizer, scheduler, metrics,
                            os.path.join(cfg.save_dir, "latest.pt"))
            if val_metrics["r2"] > metrics.best_val_r2:
                metrics.best_val_r2 = val_metrics["r2"]
                save_checkpoint(model, optimizer, scheduler, metrics,
                                os.path.join(cfg.save_dir, "best.pt"))
                print(f"  New best val R²={val_metrics['r2']:.4f}")

        if _is_distributed():
            dist.barrier()
        model.train()

    if _is_main():
        print("Training complete.")
    _cleanup_distributed()


if __name__ == "__main__":
    train()
