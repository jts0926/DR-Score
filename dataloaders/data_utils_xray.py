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


def xray_data_preparator(
    img_dir="./data/xray/",
    target="KR",
    sample_ratio=None,
    val_ratio=0.1,
    test_ratio=0.2,
    random_state=49,
):
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

    most_imgs = glob.glob(
        os.path.join(data_root["MOST"]["img_dir"], data_root["MOST"]["pattern"])
    )

    most_df = pd.DataFrame({
        "ID": [fn.split("/")[-5] for fn in most_imgs],
        "SIDE": [1 if "L" in os.path.basename(fn) else 2 for fn in most_imgs],
        "PATH": most_imgs,
    })
    most_df["ID"] = most_df["ID"].astype(str)
    most_df["SIDE"] = most_df["SIDE"].astype(int)

    most_label = pd.read_csv(data_root["MOST"]["label_csv"])
    most_label["ID"] = most_label["ID"].astype(str)
    most_label["SIDE"] = most_label["SIDE"].astype(int)
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
    MOST_df["COHORT"] = "MOST"
    MOST_df["SPLIT_ID"] = MOST_df["COHORT"] + "_" + MOST_df["ID"].astype(str)

    oai_imgs = []
    for folder in ["0.C.2_crop", "0.E.1_crop"]:
        folder_path = os.path.join(data_root["OAI"]["img_dir"], folder)
        oai_imgs.extend(
            glob.glob(os.path.join(folder_path, data_root["OAI"]["pattern"]))
        )

    OAI_img_df = pd.DataFrame({
        "ID": [fn.split("/")[-4] for fn in oai_imgs],
        "SIDE": [1 if "R" in os.path.basename(fn) else 2 for fn in oai_imgs],
        "PATH": oai_imgs,
    })
    OAI_img_df["ID"] = OAI_img_df["ID"].astype(str)
    OAI_img_df["SIDE"] = OAI_img_df["SIDE"].astype(int)

    OAI_label = pd.read_csv(data_root["OAI"]["label_csv"])
    OAI_label["ID"] = OAI_label["ID"].astype(str)
    OAI_label["SIDE"] = OAI_label["SIDE"].astype(int)
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
    OAI_df["COHORT"] = "OAI"
    OAI_df["SPLIT_ID"] = OAI_df["COHORT"] + "_" + OAI_df["ID"].astype(str)

    ment_imgs = glob.glob(
        os.path.join(data_root["MenTOR"]["img_dir"], data_root["MenTOR"]["pattern"])
    )

    MenTOR_img_df = pd.DataFrame({
        "ID": [fn.split("/")[-4] for fn in ment_imgs],
        "PATH": ment_imgs,
    })
    MenTOR_img_df["ID"] = MenTOR_img_df["ID"].astype(str)

    MenTOR_label = pd.read_csv(
        data_root["MenTOR"]["label_csv"],
        usecols=["ID", "SIDE", "event", "time"],
    ).dropna(subset=["event"])
    MenTOR_label["ID"] = MenTOR_label["ID"].astype(str)
    MenTOR_label["event"] = MenTOR_label["event"].astype(int)

    MenTOR_df = MenTOR_label.merge(MenTOR_img_df, on="ID", how="inner")
    MenTOR_df["COHORT"] = "MenTOR"
    MenTOR_df["SPLIT_ID"] = MenTOR_df["COHORT"] + "_" + MenTOR_df["ID"].astype(str)

    kick_imgs = glob.glob(
        os.path.join(data_root["KICK"]["img_dir"], data_root["KICK"]["pattern"])
    )
    kick_files = [os.path.basename(fn) for fn in kick_imgs]

    KICK_img_df = pd.DataFrame({
        "ID": [f.split("-")[0] for f in kick_files],
        "PATH": kick_imgs,
    })
    KICK_img_df["ID"] = KICK_img_df["ID"].astype(str)

    KICK_label = pd.read_csv(
        data_root["KICK"]["label_csv"],
        usecols=["ID", "SIDE", "event", "time"],
    ).dropna(subset=["event"])
    KICK_label["ID"] = "KICK" + KICK_label["ID"].astype(str)
    KICK_label["event"] = KICK_label["event"].astype(int)

    KICK_df = KICK_label.merge(KICK_img_df, on="ID", how="inner")
    KICK_df["COHORT"] = "KICK"
    KICK_df["SPLIT_ID"] = KICK_df["COHORT"] + "_" + KICK_df["ID"].astype(str)

    data_df = pd.concat([MOST_df, OAI_df], ignore_index=True)
    external_test_df = pd.concat([MenTOR_df, KICK_df], ignore_index=True)

    os.makedirs("./data/processed/", exist_ok=True)
    data_df.to_csv("./data/processed/data.csv", index=False)
    external_test_df.to_csv("./data/processed/test_external.csv", index=False)

    id_to_event = data_df.groupby("SPLIT_ID")["event"].agg(
        lambda x: x.mode()[0] if not x.mode().empty else np.nan
    )
    id_to_event = id_to_event.dropna()

    train_val_ids, test_ids = train_test_split(
        id_to_event.index,
        test_size=test_ratio,
        stratify=id_to_event.values,
        random_state=random_state,
    )

    train_val_event = id_to_event.loc[train_val_ids]
    val_size = val_ratio / (1 - test_ratio)

    train_ids, val_ids = train_test_split(
        train_val_event.index,
        test_size=val_size,
        stratify=train_val_event.values,
        random_state=random_state,
    )

    df_train = data_df[data_df["SPLIT_ID"].isin(train_ids)]
    df_val = data_df[data_df["SPLIT_ID"].isin(val_ids)]
    df_test = data_df[data_df["SPLIT_ID"].isin(test_ids)]

    df_train.to_csv("./data/processed/train.csv", index=False)
    df_val.to_csv("./data/processed/val.csv", index=False)
    df_test.to_csv("./data/processed/test.csv", index=False)

    print(f"Train shape: {df_train.shape}, Validation shape: {df_val.shape}, Test shape: {df_test.shape}, External test shape: {external_test_df.shape}")
    return df_train, df_val, df_test, external_test_df


def weighted_data_sampler(data):
    labels = data["event"].values
    classes, counts = np.unique(labels, return_counts=True)
    class_weights = {cls: sum(counts) / count for cls, count in zip(classes, counts)}
    sample_weights = [class_weights[label] for label in labels]
    return WeightedRandomSampler(sample_weights, len(labels), replacement=True)


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

        clahe = CLAHE_transform(output_format="pil")
        img = clahe(img)

        img = self.transform(img)

        label = torch.tensor(int(self.labels[idx]), dtype=torch.float32)
        time = torch.tensor(float(self.times[idx]), dtype=torch.float32)

        return img, label, time


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

        self.df_train, self.df_valid, self.df_test, self.df_external_test = xray_data_preparator(
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
        self.test_ds = XrayDataset(self.df_test, self.valid_trsfs)
        self.external_test_ds = XrayDataset(self.df_external_test, self.valid_trsfs)
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

    def test_dataloader(self):
        return [
            DataLoader(
                self.test_ds,
                batch_size=self.batch_size,
                shuffle=False,
                pin_memory=True,
            ),
            DataLoader(
                self.external_test_ds,
                batch_size=self.batch_size,
                shuffle=False,
                pin_memory=True,
            ),
        ]


if __name__ == "__main__":
    dm = XrayDataModule(
        img_dir="./data/",
        img_input_size=224,
        batch_size=16,
        target="KR",
        sample_ratio=None,
    )
    print("DataModule initialised.")