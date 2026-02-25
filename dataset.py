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
    def __init__(self, split_label="train", target_size=(512, 512), normalize=True, anatomy="macula"):
        super().__init__()
        
        self.cfg = load_config("config.json")
        self.split_label = split_label
        self.target_size = target_size
        self.normalize=normalize

        
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
        check = np.array(data_df['hvf_mtd'])
        a,b = np.percentile(check, [1,99])
        margin = (b - a) * 0.05
        a,b = a - margin, b + margin
        self.mx = a - margin
        self.mn = b + margin
        self.data = data_df[data_df["split"] == split_label]

    def rescale_label(self, label):
        normalized = (label - self.mn) / (self.mx - self.mn)
        return np.clip(normalized, 1e-6, 1 - 1e-6)

    @beartype
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        mrn = row["mrn"]
        label = torch.tensor(self.rescale_label(row['hvf_mtd']))
        oct_path = os.path.join(
            self.dcm_path, row["img_fn"].lstrip("/")
        )
        im = torch.from_numpy(pydicom.dcmread(oct_path).pixel_array).to(torch.float)
        orig_H, orig_W = im.shape[1], im.shape[2]
        
        # Resize if needed
        if self.target_size is not None:
            target_H, target_W = self.target_size
            # im shape: (frames, H, W) -> need (frames, 1, H, W) for interpolate
            im = F.interpolate(
                im.unsqueeze(1), 
                size=(target_H, target_W), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(1)
        im : Float[Tensor, "C F H W"] = einops.rearrange(im, "F H W -> 1 F H W")
        
        # Normalize to 0-1
        if self.normalize:
            im = (im - im.min()) / (im.max() - im.min() + 1e-8)
            local = {"frames": im, "label": label, "mrn": mrn}
            return local

        return im, label

    def __len__(self):
        return len(self.data)


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


