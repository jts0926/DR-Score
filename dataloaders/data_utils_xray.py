"""
data_utils_xray.py
Cleaned, privacy-safe version for manuscript supplementary materials.

This module prepares and loads knee X-ray datasets (MOST, OAI, MenTOR, KICK)
for SR-kNN Attention MIL survival prediction.

- All personal filesystem paths removed
- No selected_ids filtering
- All label CSVs use unified name: 'labels.csv'
"""

import pandas as pd
import numpy as np
import glob
import os

import torch
from torch.utils.data import Dataset, WeightedRandomSampler, DataLoader
from torchvision import transforms
from pytorch_lightning import LightningDataModule
from PIL import Image

from sklearn.model_selection import train_test_split

from .transforms import *


# =========================================================================
# Helper: Prepare Multi-Cohort X-ray Survival Dataset
# =========================================================================

def xray_data_preparator(
    img_dir="./data/xray/",
    target="KR",
    sample_ratio=None,
    val_ratio=0.2,
    random_state=49,
):
    """
    Loads and merges MOST, OAI, MenTOR, and KICK datasets.

    Assumed folder structure (can be adapted as needed):
    ./data/
        MOST/
            images/            # PNG files
            labels.csv         # MUST contain columns: ID, SIDE, event, time
        OAI/
            images/
            labels.csv         # MUST contain KR_DAYS_FROM_VISIT, KR_in108m, ID, SIDE
        MenTOR/
            images/
            labels.csv         # MUST contain: ID, SIDE, event, time
        KICK/
            images/
            labels.csv         # MUST contain: ID, SIDE, event, time

    No selected_ids filtering is applied in this cleaned version.
    """

    # ---------------------------------------------------------------------
    # Define all dataset directories (unified label file name: labels.csv)
    # ---------------------------------------------------------------------
    data_root = {
        "MOST": {
            "img_dir": "./data/MOST/images/",
            "label_csv": "./data/MOST/labels.csv",
            "pattern": "*/*/*/*/*/*.png",
        },
        "OAI": {
            "img_dir": "./data/OAI/images/",
            "label_csv": "./data/OAI/labels.csv",
            "pattern": "*/*/*/*.png",
        },
        "MenTOR": {
            "img_dir": "./data/MenTOR/images/",
            "label_csv": "./data/MenTOR/labels.csv",
            "pattern": "*/*/*/*.png",
        },
        "KICK": {
            "img_dir": "./data/KICK/images/",
            "label_csv": "./data/KICK/labels.csv",
            "pattern": "*.png",
        },
    }

    # ===============================
    # Load MOST
    # ===============================
    most_imgs = glob.glob(
        os.path.join(data_root["MOST"]["img_dir"], data_root["MOST"]["pattern"])
    )

    most_df = pd.DataFrame({
        "ID": [fn.split("/")[-5] for fn in most_imgs],
        "SIDE": [1 if "L" in os.path.basename(fn) else 2 for fn in most_imgs],
        "PATH": most_imgs,
    })
    most_df["SIDE"] = most_df["SIDE"].astype(int)

    most_label = pd.read_csv(data_root["MOST"]["label_csv"])
    # Assumes 'time' (in months) and 'event' are already present
    most_label["time"] = np.where(
        most_label["time"] > 0,
        most_label["time"],
        np.nan,
    ).astype(float)
    most_label["event"] = most_label["event"].astype(int)
    max_days = most_label["time"].max()
    most_label["time"].fillna(max_days + 1, inplace=True)
    most_label["time"] = most_label["time"].astype(int)

    MOST_df = most_label.merge(most_df, on=["ID", "SIDE"], how="inner")

    # ===============================
    # Load OAI
    # ===============================
    oai_imgs = []
    # If you have multiple folders under OAI/images, you can adjust here
    for folder in ["0.C.2_crop", "0.E.1_crop"]:
        folder_path = os.path.join(data_root["OAI"]["img_dir"], folder)
        oai_imgs.extend(
            glob.glob(os.path.join(folder_path, data_root["OAI"]["pattern"]))
        )

    OAI_img_df = pd.DataFrame({
        "ID": [fn.split("/")[-4] for fn in oai_imgs],
        "SIDE": [1 if "R" in os.path.basename(fn) else 2 for fn in oai_imgs],
        "PATH": oai_imgs,
    }).astype({"ID": int, "SIDE": int})

    OAI_label = pd.read_csv(data_root["OAI"]["label_csv"])
    # Assumes OAI labels contain 'KR_DAYS_FROM_VISIT' and 'KR_in108m'
    OAI_label["time"] = np.where(
        OAI_label["KR_DAYS_FROM_VISIT"] > 0,
        np.floor(OAI_label["KR_DAYS_FROM_VISIT"] / 30.4),
        np.nan,
    ).astype("Int32")
    OAI_label["event"] = OAI_label["KR_in108m"].astype(int)
    OAI_label.drop(["KR_DAYS_FROM_VISIT", "KR_in108m"], axis=1, inplace=True)
    OAI_label.dropna(subset=["event"], inplace=True)
    OAI_label["time"].fillna(max_days + 1, inplace=True)

    OAI_df = OAI_label.merge(OAI_img_df, on=["ID", "SIDE"], how="inner")

    # ===============================
    # Load MenTOR
    # ===============================
    ment_imgs = glob.glob(
        os.path.join(data_root["MenTOR"]["img_dir"], data_root["MenTOR"]["pattern"])
    )

    MenTOR_img_df = pd.DataFrame({
        "ID": [fn.split("/")[-4] for fn in ment_imgs],
        "PATH": ment_imgs,
    })

    MenTOR_label = pd.read_csv(
        data_root["MenTOR"]["label_csv"],
        usecols=["ID", "SIDE", "event", "time"],
    ).dropna(subset=["event"])

    MenTOR_df = MenTOR_label.merge(MenTOR_img_df, on="ID", how="inner")

    # ===============================
    # Load KICK
    # ===============================
    kick_imgs = glob.glob(
        os.path.join(data_root["KICK"]["img_dir"], data_root["KICK"]["pattern"])
    )
    kick_files = [os.path.basename(fn) for fn in kick_imgs]

    KICK_img_df = pd.DataFrame({
        "ID": [f.split("-")[0] for f in kick_files],
        "PATH": kick_imgs,
    })

    KICK_label = pd.read_csv(
        data_root["KICK"]["label_csv"],
        usecols=["ID", "SIDE", "event", "time"],
    ).dropna(subset=["event"])
    KICK_label["ID"] = "KICK" + KICK_label["ID"].astype(str)

    KICK_df = KICK_label.merge(KICK_img_df, on="ID", how="inner")

    # =========================================================================
    # Merge MOST + OAI (you can extend to MenTOR / KICK as needed)
    # =========================================================================
    data_df = pd.concat([MOST_df, OAI_df], ignore_index=True)
    # external_val_df = pd.concat([MenTOR_df, KICK_df], ignore_index=True)
    
    os.makedirs("./data/processed/", exist_ok=True)
    data_df.to_csv("./data/processed/data.csv", index=False)

    # =========================================================================
    # Train/Validation Split (stratified by event at ID level)
    # =========================================================================
    id_to_event = data_df.groupby("ID")["event"].agg(
        lambda x: x.mode()[0] if not x.mode().empty else np.nan
    )
    id_to_event = id_to_event.dropna()

    train_ids, val_ids = train_test_split(
        id_to_event.index,
        test_size=val_ratio,
        stratify=id_to_event.values,
        random_state=random_state,
    )

    df_train = data_df[data_df["ID"].isin(train_ids)]
    df_val = data_df[data_df["ID"].isin(val_ids)]
    # df_val = external_val_df

    df_train.to_csv("./data/processed/train.csv", index=False)
    df_val.to_csv("./data/processed/val.csv", index=False)
    # df_val.to_csv("./data/processed/val_external_menTOR_KICK.csv", index=False)

    print(f"Train shape: {df_train.shape}, Validation shape: {df_val.shape}")
    return df_train, df_val


# =========================================================================
# Weighted Sampler
# =========================================================================

def weighted_data_sampler(data):
    """Return WeightedRandomSampler for imbalanced event labels."""
    labels = data["event"].values
    classes, counts = np.unique(labels, return_counts=True)
    class_weights = [sum(counts) / c for c in counts]
    sample_weights = [class_weights[label] for label in labels]
    return WeightedRandomSampler(sample_weights, len(labels), replacement=True)


# =========================================================================
# Dataset Class
# =========================================================================

class XrayDataset(Dataset):
    def __init__(self, df, trsf):
        self.transform = trsf
        self.paths = df["PATH"].tolist()
        self.labels = df["event"].tolist()
        self.times = df["time"].tolist()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = transforms.functional.rgb_to_grayscale(img)

        # CLAHE contrast enhancement
        clahe = CLAHE_transform(output_format="pil")
        img = clahe(img)

        img = self.transform(img)

        label = torch.tensor(int(self.labels[idx]), dtype=torch.float32)
        time = torch.tensor(float(self.times[idx]), dtype=torch.float32)

        return img, label, time


# =========================================================================
# Lightning DataModule
# =========================================================================

class XrayDataModule(LightningDataModule):
    def __init__(
        self,
        img_dir,
        img_input_size,
        batch_size,
        target,
        sample_ratio,
        random_state=40,
        weighted_sampling=False,
        ssl_training=False,
    ):
        super().__init__()

        self.img_dir = img_dir
        self.img_input_size = img_input_size
        self.batch_size = batch_size
        self.target = target
        self.sample_ratio = sample_ratio
        self.random_state = random_state
        self.weighted_sampling = weighted_sampling

        self.df_train, self.df_valid = xray_data_preparator(
            img_dir=self.img_dir,
            target=self.target,
            sample_ratio=self.sample_ratio,
            random_state=self.random_state,
        )

        if ssl_training:
            self.train_trsfs = GetBasicTransforms(
                (self.img_input_size, self.img_input_size)
            )
        else:
            self.train_trsfs = GetTrainTransforms(
                (self.img_input_size, self.img_input_size)
            )
        self.valid_trsfs = GetValidTransforms(
            (self.img_input_size, self.img_input_size)
        )

    def setup(self, stage=None):
        self.train_ds = XrayDataset(self.df_train, self.train_trsfs)
        self.valid_ds = XrayDataset(self.df_valid, self.valid_trsfs)
        self.sampler = (
            weighted_data_sampler(self.df_train) if self.weighted_sampling else None
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=(self.sampler is None),
            sampler=self.sampler,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_ds,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
        )


if __name__ == "__main__":
    dm = XrayDataModule(
        img_dir="./data/",
        img_input_size=224,
        batch_size=16,
        target="KR",
        sample_ratio=None,
    )
    print("DataModule initialised.")