import torch
import torch.nn as nn
import einops as ei
import math


def create_patchbag(img, patch_height, patch_width, color_img=False):
    """
    Parameters
    ----------
    img: For greylevel image: B x D x H x W; B: batch size; D: dimension (no. of slides); H: height; W: width
         For 2D color image: B x C x H x W; B: batch size; C: n_channel (no. of color channels); H: height; W: width
    Returns
    -------
    num_patches: number of patches (int)
    img_rg:
        For greylevel image:
        image rearrangement operation for nn.Sequential() -- (B x Patch_num x no. of slides) x C x p1 x p2; Patch_num = (H x W)//(p1 x p2)
        # c=1: empty dimension as place holder for image channel

        For color image:
        image rearrangement operation for nn.Sequential() -- (B x Patch_num) x C x p1 x p2; Patch_num = (H x W)//(p1 x p2)
        # C=3: image RGB color channel
    """
    _, _, img_height, img_width = img.size()
    assert (img_height % patch_height == 0) and (img_width % patch_width == 0)

    if color_img:
        img_rg = ei.rearrange(img,
                              'B C (h p1) (w p2) -> (B h w) C p1 p2',
                              p1=patch_height, p2=patch_width)
    else:
        img_rg = ei.rearrange(img,
                              'B D (h p1) (w p2) -> (B D h w) 1 p1 p2',
                              p1=patch_height, p2=patch_width)
    return img_rg


def pos_encoding_1d(B, N, L):
    """
    Parameters
    ----------
    L: dimension of the patch feature vector
    N: number of patches
    B: Batch size
    Returns
    -------
    pe: postition encodings (B x N x L)
    """
    if L % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(L))
    pe = torch.zeros(N, L) # Create positional encoding matrix for each input
    position = torch.arange(0, N).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, L, 2, dtype=torch.float) * -(math.log(10000.0) / L)))
    pe[:, 0::2] = torch.sin(position.float() * div_term) # N x L
    pe[:, 1::2] = torch.cos(position.float() * div_term) # N x L
    pe_batch = pe.repeat(B, 1, 1) # B x N x L
    return pe_batch.to('cuda:0')


def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe


class ResidualConnection(nn.Module):
    """
    Residual summation followed by instance norm and dropout
    """
    def __init__(self, feat_dim):
        super().__init__()
        #self.norm = nn.InstanceNorm1d(feat_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, h):
        return x + self.dropout(h)
