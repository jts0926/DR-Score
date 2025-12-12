"""
SRKNN Attention Multiple Instance Learning (MIL) Model
Clean version for manuscript supplementary code.

- Personal paths removed
- Safe, generic output paths used
- No changes to model functionality
"""

import os
import datetime
from csv import writer
import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics as tm

from tqdm import tqdm
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score
)

from loss_func import *
from .SRKNNabmil_arch import *
from .base_model import *

import deepsurv
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
from sksurv.metrics import cumulative_dynamic_auc


# ---------------------------------------------------------
# Utility: Concordance Index
# ---------------------------------------------------------
def concordance_index(y_true_time, y_pred, y_true_event):
    """
    Simple implementation of C-index for survival risk scores.
    Higher y_pred indicates higher risk.
    """
    n = len(y_true_time)
    num_correct, num_comparable = 0, 0

    for i in range(n):
        for j in range(i + 1, n):

            # Comparable pair if either event occurred
            if y_true_event[i] == 1 or y_true_event[j] == 1:
                num_comparable += 1

                if (y_true_time[i] < y_true_time[j] and y_pred[i] > y_pred[j]) or \
                   (y_true_time[i] > y_true_time[j] and y_pred[i] < y_pred[j]):
                    num_correct += 1

                elif y_true_time[i] == y_true_time[j] and y_pred[i] == y_pred[j]:
                    num_correct += 0.5

    return num_correct / num_comparable if num_comparable > 0 else 0


# ---------------------------------------------------------
# Spatial L1 Distance Matrix for kNN Attention
# ---------------------------------------------------------
def spatial_distance_mat(img_shape, patch_size):
    """
    Parameters
    ----------
    img_shape : tuple
        (num_slices, height, width)
    patch_size : int

    Returns
    -------
    dist_mat : torch.Tensor
        Matrix of L1 distances between all patch coordinates.
    """
    S, H, W = img_shape
    h = H // patch_size
    w = W // patch_size

    # Patch coordinates: (slice, row, col)
    patch_coords = np.array([
        [i, j, k]
        for i in range(S)
        for j in range(h)
        for k in range(w)
    ])

    dist_mat = []
    for coord in tqdm(patch_coords, desc="Computing spatial distances"):
        dist = np.linalg.norm(patch_coords - coord, ord=1, axis=1)
        dist_mat.append(dist)

    return torch.tensor(np.array(dist_mat), dtype=torch.float32)


# ---------------------------------------------------------
# Main MIL Model Wrapper
# ---------------------------------------------------------
class SRkNNAttentionMIL(BaseModel):
    """
    Wrapper class for SRkNN Attention MIL.
    """

    def __init__(self,
                 color_img=False,
                 num_classes=1,
                 loss='deepsurv',
                 lr=0.001,
                 patch_size=32,
                 img_dim=3,
                 pos_encoding=True,
                 extraction_layer='both',
                 hist_output_size=10,
                 ssl_feat_pretrain_fname=None,
                 feat_embedding_size=200,
                 att_embedding_size=128,
                 knn_att_type='representational',
                 topk_R=100,
                 topk_S=2,
                 spatial_dist_mat=None,
                 training=True):

        super().__init__(num_classes, loss, lr)

        self.lr = lr
        self.loss = loss
        self.patch_size = patch_size
        self.extraction_layer = extraction_layer
        self.hist_output_size = hist_output_size
        self.feat_embedding_size = feat_embedding_size
        self.att_embedding_size = att_embedding_size
        self.knn_att_type = knn_att_type
        self.topk_R = topk_R
        self.topk_S = topk_S

        # Model architecture
        self.model = SRkNNAttentionMIL_(
            ssl_feat_pretrain_fname=ssl_feat_pretrain_fname,
            color_img=color_img,
            patch_size=self.patch_size,
            img_dim=img_dim,
            pos_encoding=pos_encoding,
            extraction_layer=extraction_layer,
            hist_output_size=hist_output_size,
            feat_embedding_size=feat_embedding_size,
            att_embedding_size=att_embedding_size,
            knn_att_type=knn_att_type,
            topk_R=topk_R,
            topk_S=topk_S,
            spatial_dist_mat=spatial_dist_mat,
            training=training
        ).to("cuda:0")

    # -----------------------------------------------------
    # Forward pass
    # -----------------------------------------------------
    def forward(self, x):
        h = self.model(x)
        self.Att = self.model.a
        self.RkNNAtt = self.model.RkNNAtt
        self.SkNNAtt = self.model.SkNNAtt
        return h

    # -----------------------------------------------------
    # Save / Load
    # -----------------------------------------------------
    def save(self, fname):
        os.makedirs("./trained_models", exist_ok=True)
        torch.save(self.model.state_dict(), f"./trained_models/{fname}.pth")

    def load(self, fname):
        path = f"./trained_models/{fname}.pth"
        self.model.load_state_dict(torch.load(path), strict=False)
        self.model.to("cuda:0")
        return self

    # -----------------------------------------------------
    # Test / Evaluation
    # -----------------------------------------------------
    def test(self, dloader, task, model_name, save=True):
        """
        Evaluate model on validation/test set.
        Computes:
        - Predictions
        - Concordance Index
        - Time-dependent AUC
        """

        y_preds, y_trues = [], []

        self.model.to("cuda:0")
        self.model.eval()

        with torch.no_grad():
            for img, event, time in dloader:
                img = img.to("cuda:0")
                time = time.to("cuda:0")
                event = event.to("cuda:0")

                pred = self.model(img)
                y_preds.append(pred.cpu().numpy())
                y_trues.append(
                    np.column_stack((
                        time.cpu().numpy(),
                        event.cpu().numpy()
                    ))
                )

        y_preds = np.concatenate(y_preds)
        y_trues = np.concatenate(y_trues)

        if save:
            os.makedirs("./outputs", exist_ok=True)
            np.savetxt(
                "./outputs/model_predictions.csv",
                np.column_stack((y_preds, y_trues)),
                delimiter=",",
                header="pred,time,event",
                comments=""
            )

        # ----------------------------
        # Compute C-index
        # ----------------------------
        c_index = concordance_index(
            y_true_time=y_trues[:, 0],
            y_pred=y_preds,
            y_true_event=y_trues[:, 1]
        )
        print("C-index:", c_index)

        # ----------------------------
        # Time-dependent AUC
        # ----------------------------
        y_struct = np.array(
            [(bool(e), t) for t, e in zip(y_trues[:, 0], y_trues[:, 1])],
            dtype=[("event", bool), ("time", float)]
        )

        times = np.arange(3, 108, 3)
        auc, mean_auc = cumulative_dynamic_auc(y_struct, y_struct, y_preds, times)
        print("Mean AUC:", mean_auc)

        # Placeholder risk groups (if user wants)
        low_risk_idx = medium_risk_idx = high_risk_idx = None

        return y_preds, y_trues, low_risk_idx, medium_risk_idx, high_risk_idx