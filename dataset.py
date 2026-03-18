import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import Tensor
import numpy as np
import os
import pydicom
from jaxtyping import Float
from beartype import beartype
import einops
from utils import load_config


class HVFDataset(torch.utils.data.Dataset):
    def __init__(self, split_label="train", target_size=(512, 512), normalize=True, anatomy="macula", center_crop_frac=None, num_frames=None, norm_mode="zscore"):
        super().__init__()

        self.cfg = load_config("config.json")
        self.split_label = split_label
        self.target_size = target_size
        self.normalize=normalize
        self.center_crop_frac = center_crop_frac  # e.g. 0.5 keeps center 50% of W
        self.num_frames = num_frames  # resample temporal dim if set (e.g. 128)
        self.norm_mode = norm_mode  # "zscore" or "minmax"

        
        self.hvfmd_path = "macula_oct_partially_deduplicated.tsv" if anatomy == "macula" else "optic_nerve_oct_partially_deduplicated.tsv"
        self.dcm_path = self.cfg.dcm_path

        data_df = pd.read_csv(self.hvfmd_path, sep="\t")

        if os.path.exists("training_splits.csv"):
            print("Found existing splits")
            splits = pd.read_csv("training_splits.csv")
        else:
            print("No splits exist, creating...")
            pids = data_df["mrn"].unique()
            indices = np.arange(len(pids))
            np.random.shuffle(indices)
            shuffled_mrns = pids[indices]

            n = len(shuffled_mrns)
            train_end = int(0.7 * n)
            val_end = int(0.85 * n)

            split_dict = {}
            split_dict.update({mrn: "train" for mrn in shuffled_mrns[:train_end]})
            split_dict.update({mrn: "val" for mrn in shuffled_mrns[train_end:val_end]})
            split_dict.update({mrn: "test" for mrn in shuffled_mrns[val_end:]})

            splits = pd.DataFrame(
                [
                    {"mrn": mrn, "split": split}
                    for mrn, split in split_dict.items()
                ]
            )
            splits.to_csv("training_splits.csv", index=False)

        data_df = pd.merge(data_df, splits, left_on="hvf_mrn", right_on="mrn")
        all_mtd = np.array(data_df['hvf_mtd'])
        # Z-score stats (always computed for unscale_label)
        self.label_mean = float(np.mean(all_mtd))
        self.label_std = float(np.std(all_mtd))
        # Min-max stats
        p1, p99 = np.percentile(all_mtd, [1, 99])
        margin = (p99 - p1) * 0.05
        self.label_min = p1 - margin
        self.label_max = p99 + margin
        self.data = data_df[data_df["split"] == split_label]

    def rescale_label(self, label):
        if self.norm_mode == "minmax":
            normalized = (label - self.label_min) / (self.label_max - self.label_min)
            return np.clip(normalized, 1e-6, 1 - 1e-6)
        return (label - self.label_mean) / self.label_std

    def unscale_label(self, z):
        """Convert normalized label back to original MTD scale."""
        if self.norm_mode == "minmax":
            return z * (self.label_max - self.label_min) + self.label_min
        return z * self.label_std + self.label_mean

    def _load_and_preprocess(self, row):
        """Load DICOM, crop, resize, resample — the expensive part."""
        oct_path = os.path.join(
            self.dcm_path, row["img_fn"].lstrip("/")
        )
        im = torch.from_numpy(pydicom.dcmread(oct_path).pixel_array).to(torch.float)
        # im shape: (frames, H, W)

        # Center crop along W (left-right) before resize — keeps center of each B-scan
        if self.center_crop_frac is not None:
            W = im.shape[2]
            crop_w = int(W * self.center_crop_frac)
            start = (W - crop_w) // 2
            im = im[:, :, start:start + crop_w]

        # Resize if needed
        if self.target_size is not None:
            target_H, target_W = self.target_size
            im = F.interpolate(
                im.unsqueeze(1),
                size=(target_H, target_W),
                mode='bilinear',
                align_corners=False
            ).squeeze(1)
        # Resample temporal dimension if num_frames is set
        if self.num_frames is not None and im.shape[0] != self.num_frames:
            im = F.interpolate(
                im.unsqueeze(0).unsqueeze(0),
                size=(self.num_frames, im.shape[1], im.shape[2]),
                mode='trilinear',
                align_corners=False,
            ).squeeze(0).squeeze(0)

        im : Float[Tensor, "C F H W"] = einops.rearrange(im, "F H W -> 1 F H W")

        # Normalize to 0-1
        if self.normalize:
            im = (im - im.min()) / (im.max() - im.min() + 1e-8)

        return im

    @beartype
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        mrn = row["mrn"]
        label = torch.tensor(self.rescale_label(row['hvf_mtd']))
        im = self._load_and_preprocess(row)

        if self.normalize:
            return {"frames": im, "label": label, "mrn": mrn}
        return im, label

    def __len__(self):
        return len(self.data)

    def test_retest_variability(self, n_bins=10):
        """Calculate test-retest variability for patients with multiple samples.

        Groups by MRN, computes within-patient std of normalized labels,
        then reports overall and per-bin statistics.

        Returns a dict with:
          - overall: {n_patients, n_samples, mean_std, median_std, mean_range, median_range}
          - bins: list of per-bin dicts with {bin_center, bin_lo, bin_hi, n_patients, mean_std, ...}
          - raw: DataFrame with per-patient stats
        """
        df = self.data.copy()
        df["label_norm"] = df["hvf_mtd"].apply(self.rescale_label)

        grouped = df.groupby("mrn")["label_norm"]
        # Only patients with ≥2 samples
        multi = grouped.filter(lambda x: len(x) >= 2)
        if len(multi) == 0:
            print("No patients with multiple samples found.")
            return None

        multi_grouped = df.loc[multi.index].groupby("mrn")["label_norm"]
        per_patient = pd.DataFrame({
            "count": multi_grouped.count(),
            "mean": multi_grouped.mean(),
            "std": multi_grouped.std(),
            "range": multi_grouped.apply(lambda x: x.max() - x.min()),
        })

        # Overall label distribution stats (all samples, not just repeated)
        all_labels = df["label_norm"].values
        label_stats = {
            "n_total_samples": len(all_labels),
            "label_std": float(np.std(all_labels)),
            "label_mean": float(np.mean(all_labels)),
            "label_median": float(np.median(all_labels)),
            "label_min": float(np.min(all_labels)),
            "label_max": float(np.max(all_labels)),
            "label_iqr": float(np.percentile(all_labels, 75) - np.percentile(all_labels, 25)),
        }

        # Overall test-retest stats (patients with ≥2 samples)
        overall = {
            **label_stats,
            "n_patients": len(per_patient),
            "n_samples": int(per_patient["count"].sum()),
            "mean_std": float(per_patient["std"].mean()),
            "median_std": float(per_patient["std"].median()),
            "mean_range": float(per_patient["range"].mean()),
            "median_range": float(per_patient["range"].median()),
        }

        print(f"\n{'='*60}")
        print(f"Label Distribution (z-score scale)")
        print(f"{'='*60}")
        print(f"Total samples: {label_stats['n_total_samples']}")
        print(f"Label std:     {label_stats['label_std']:.4f}")
        print(f"Label mean:    {label_stats['label_mean']:.4f}  "
              f"median: {label_stats['label_median']:.4f}")
        print(f"Label range:   [{label_stats['label_min']:.4f}, {label_stats['label_max']:.4f}]  "
              f"IQR: {label_stats['label_iqr']:.4f}")

        print(f"\n{'='*60}")
        print(f"Test-Retest Variability (z-score scale)")
        print(f"{'='*60}")
        print(f"Patients with ≥2 samples: {overall['n_patients']} "
              f"({overall['n_samples']} total samples)")
        print(f"Within-patient label std:    mean={overall['mean_std']:.4f}  "
              f"median={overall['median_std']:.4f}")
        print(f"Within-patient label range:  mean={overall['mean_range']:.4f}  "
              f"median={overall['median_range']:.4f}")

        # Binned by patient mean label (data-driven edges)
        all_means = per_patient["mean"].values
        edges = np.linspace(all_means.min(), all_means.max(), n_bins + 1)
        bins_out = []
        print(f"\n{'Bin':>12s}  {'n_pts':>5s}  {'mean_std':>8s}  {'med_std':>8s}  "
              f"{'mean_rng':>8s}  {'med_rng':>8s}")
        print("-" * 60)
        for i in range(n_bins):
            lo, hi = edges[i], edges[i + 1]
            if i == n_bins - 1:
                mask = (per_patient["mean"] >= lo) & (per_patient["mean"] <= hi)
            else:
                mask = (per_patient["mean"] >= lo) & (per_patient["mean"] < hi)
            subset = per_patient[mask]
            center = (lo + hi) / 2
            bin_info = {
                "bin_center": float(center),
                "bin_lo": float(lo),
                "bin_hi": float(hi),
                "n_patients": len(subset),
            }
            if len(subset) > 0:
                bin_info.update({
                    "mean_std": float(subset["std"].mean()),
                    "median_std": float(subset["std"].median()),
                    "mean_range": float(subset["range"].mean()),
                    "median_range": float(subset["range"].median()),
                })
            else:
                bin_info.update({
                    "mean_std": 0.0, "median_std": 0.0,
                    "mean_range": 0.0, "median_range": 0.0,
                })
            bins_out.append(bin_info)
            print(f"  [{lo:.2f},{hi:.2f})  {bin_info['n_patients']:>5d}  "
                  f"{bin_info['mean_std']:>8.4f}  {bin_info['median_std']:>8.4f}  "
                  f"{bin_info['mean_range']:>8.4f}  {bin_info['median_range']:>8.4f}")

        print(f"{'='*60}\n")

        return {"overall": overall, "bins": bins_out, "raw": per_patient}


class FeatureDataset(torch.utils.data.Dataset):
    """Loads pre-extracted (N, 1024) features and (N,) labels from .npy files."""

    def __init__(self, feature_path: str, label_path: str):
        self.features = np.load(feature_path)
        self.labels = np.load(label_path)
        assert len(self.features) == len(self.labels), (
            f"Feature/label length mismatch: {len(self.features)} vs {len(self.labels)}"
        )
        print(f"FeatureDataset: {len(self)} samples from {feature_path}")

    def __getitem__(self, idx):
        return {
            "features": torch.from_numpy(self.features[idx]),
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

    def __len__(self):
        return len(self.labels)


