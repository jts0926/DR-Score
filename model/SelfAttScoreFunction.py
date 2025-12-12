import torch
import torch.nn as nn
import einops as ei


def ScalarSA(x, norm=True):
    """
    Parameters
    ----------
    x: feature vectors -- N x L
    norm: if True, normalized dot product (cosine similarity), else simple dot product
    Returns
    -------
    score: similarity score matrix between each patch -- N x N
    """
    if norm:
        N, L = x.size()
        x_n = x.norm(dim=1)[:, None]  # Calculate norm of each vector
        x_norm = x / torch.max(x_n, 1e-8 * torch.ones_like(x_n))  # Create unit vectors
        score = torch.matmul(x_norm, torch.transpose(x_norm, 0, 1))  # calculate cosine similarity N x N
    else:
        score = torch.matmul(x, torch.transpose(x, 0, 1))  # calculate cosine similarity N x N

    return score

