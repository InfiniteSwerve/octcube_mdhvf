"""
End-to-end fine-tuning of OCTCube for HVF MTD regression.

All encoder layers unfrozen with layer-wise learning rate decay (LLRD).
Linear head with dropout, matching the official OCTCubeM fine-tuning recipe.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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
    eff_batch = cfg.batch_size * cfg.grad_accum_steps
    lr = cfg.base_lr * eff_batch / 256
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
    def __init__(self):
        self.data = {}
        for split in ["train", "val"]:
            self.data[split] = {"iterations": [], "metrics": defaultdict(list)}
        self.opt_step = 0
        self.epoch = 0
        self.rolling_preds = deque(maxlen=2000)
        self.rolling_gts = deque(maxlen=2000)
        self.eta_str = ""

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
            return {"mae": 0, "pearson_r": 0}
        p = np.array(self.rolling_preds)
        g = np.array(self.rolling_gts)
        mae = np.abs(p - g).mean()
        r = np.corrcoef(p, g)[0, 1] if np.std(p) > 1e-8 else 0.0
        return {"mae": float(mae), "pearson_r": float(r)}

    def plot(self):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        for split, color in [("train", "C0"), ("val", "C1")]:
            iters = self.data[split]["iterations"]
            for ax, key in zip(axes.flat, ["loss", "mae", "pearson_r", "lr"]):
                vals = self.data[split]["metrics"].get(key, [])
                if vals and len(vals) == len(iters):
                    ax.plot(iters, vals, color=color, alpha=0.6, label=split, linewidth=0.8)
                    ax.set_title(key)
                    ax.legend(fontsize=7)
        fig.suptitle(f"Epoch {self.epoch} | step {self.opt_step} | {self.eta_str}", fontsize=10)
        fig.tight_layout()
        plt.savefig("finetune_metrics.png", dpi=120)
        plt.close()

    def plot_scatter(self):
        if len(self.rolling_preds) < 20:
            return
        p = np.array(self.rolling_preds)
        g = np.array(self.rolling_gts)
        mae = np.abs(p - g).mean()
        r = np.corrcoef(p, g)[0, 1] if np.std(p) > 1e-8 else 0.0
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(g, p, s=4, alpha=0.4)
        lo, hi = min(g.min(), p.min()), max(g.max(), p.max())
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1)
        ax.set_xlabel("GT")
        ax.set_ylabel("Pred")
        ax.set_title(f"Rolling (n={len(p)}) MAE={mae:.4f} r={r:.4f}")
        fig.tight_layout()
        plt.savefig("finetune_scatter.png", dpi=120)
        plt.close()

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
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        pred = model(images.cuda()).float()
        loss = F.mse_loss(pred, labels.cuda().float())
    (loss / accum).backward()
    return {"loss": loss.item(), "pred_std": pred.detach().std().item()}, pred.detach()


def validate(model, loader, cfg: Config):
    model.eval()
    all_preds, all_labels = [], []
    total_loss, n = 0.0, 0
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for batch in loader:
            if n >= cfg.val_max_volumes:
                break
            imgs = batch["frames"].cuda()
            labels = batch["label"].cuda()
            pred = model(imgs).float()
            total_loss += F.mse_loss(pred, labels.float()).item()
            all_preds.append(pred.cpu())
            all_labels.append(batch["label"])
            n += imgs.shape[0]
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
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "opt_step": metrics.opt_step,
        "epoch": metrics.epoch,
    }, path)
    print(f"Saved checkpoint to {path}")


def train():
    cfg = Config()
    os.makedirs(cfg.save_dir, exist_ok=True)
    os.makedirs(cfg.scatter_dir, exist_ok=True)

    # Data
    ds_kwargs = dict(
        target_size=(cfg.img_size, cfg.img_size),
        normalize=True,
        center_crop_frac=cfg.center_crop_frac,
    )
    train_loader = torch.utils.data.DataLoader(
        HVFDataset(split_label="train", **ds_kwargs),
        batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        HVFDataset(split_label="val", **ds_kwargs),
        batch_size=cfg.batch_size, num_workers=cfg.num_workers,
    )

    # Model
    model = OCTCubeFinetune(cfg).cuda()

    # Optimizer with LLRD
    param_groups, lr = build_param_groups(model.encoder, model.head, cfg)
    optimizer = torch.optim.AdamW(param_groups)

    # Scheduler
    opt_steps_per_epoch = len(train_loader) // cfg.grad_accum_steps
    scheduler = build_scheduler(optimizer, cfg, opt_steps_per_epoch)

    metrics = Metrics()
    accum = cfg.grad_accum_steps

    print("=" * 60)
    print(f"Fine-tuning OCTCube-{cfg.model_size} end-to-end")
    print(f"  img_size={cfg.img_size}, batch={cfg.batch_size}, accum={accum}")
    print(f"  epochs={cfg.epochs}, warmup={cfg.warmup_epochs}")
    print(f"  base_lr={cfg.base_lr}, eff_lr={lr:.6f}, layer_decay={cfg.layer_decay}")
    print(f"  weight_decay={cfg.weight_decay}, dropout={cfg.dropout}")
    print(f"  {len(train_loader)} batches/epoch, {opt_steps_per_epoch} opt steps/epoch")
    print("=" * 60)

    for epoch in range(1, cfg.epochs + 1):
        metrics.epoch = epoch
        model.train()
        optimizer.zero_grad()

        accum_loss = 0.0
        accum_preds, accum_labels = [], []
        accum_count = 0
        epoch_start = time.time()
        total_batches = len(train_loader)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs}", leave=True)
        for batch_idx, batch in enumerate(pbar):
            step_metrics, preds = forward_step(
                batch["frames"], batch["label"], model, accum,
            )
            accum_loss += step_metrics["loss"]
            accum_preds.append(preds)
            accum_labels.append(batch["label"])
            accum_count += 1

            pbar.set_postfix_str(
                f"micro={accum_count}/{accum} loss={step_metrics['loss']:.4f}", refresh=False
            )

            is_boundary = (batch_idx + 1) % accum == 0 or (batch_idx + 1) == total_batches
            if is_boundary:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                metrics.opt_step += 1

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

                if metrics.opt_step % cfg.plot_interval == 0:
                    metrics.plot()
                    metrics.plot_scatter()

                if metrics.opt_step % cfg.val_interval == 0:
                    val_metrics, _, _ = validate(model, val_loader, cfg)
                    metrics.append("val", val_metrics)
                    metrics.plot()
                    model.train()

        # End-of-epoch validation
        val_metrics, val_preds, val_gts = validate(model, val_loader, cfg)
        metrics.append("val", val_metrics)
        print(f"Epoch {epoch}/{cfg.epochs}  val_loss={val_metrics['loss']:.5f}  "
              f"mae={val_metrics['mae']:.4f}  r={val_metrics['pearson_r']:.4f}  "
              f"r2={val_metrics['r2']:.4f}")

        metrics.plot()
        metrics.plot_scatter()
        metrics.save(os.path.join(cfg.save_dir, "metrics.json"))
        save_checkpoint(model, optimizer, scheduler, metrics,
                        os.path.join(cfg.save_dir, "latest.pt"))
        model.train()

    print("Training complete.")


if __name__ == "__main__":
    train()
