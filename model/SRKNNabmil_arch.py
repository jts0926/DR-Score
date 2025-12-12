import torch
import torch.nn as nn
import torch.nn.functional as F
import einops as ei
import deepsurv
from torchvision.transforms.functional import rgb_to_grayscale
import numpy as np
import os

from .conv_feat_extractor import *
from .handcrafted_feat_extractor import *
from .auxilliary_layer import *


class SRkNNAttentionMIL_(nn.Module):
    def __init__(self,
                 color_img=False,
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
        super(SRkNNAttentionMIL_, self).__init__()
        """
        :color_img: True if the image is RGB
        :param patch_size: side length of a square patch
        :img_dim: CT-scans: 3; X-ray: 2
        :param extraction_layer: 'both', 'conv', or 'hist'
        :param hist_embedding_size (L1): output vector length of the histogram feature extractor
        :param conv_embedding_size (L2): output vector length of the conv feature extractor
        :param att_embedding_size (D): embedding vector length of the attention layer
        :param knn_att_type: 'representational', 'spatial', 'both', 'none', 'full'
        :param topk_R: patch representational top-k nearest neighbours
        :param topk_S: patch spatial top-k nearest neighbours
        :param spatial_dist_mat: precalculated distance matrix for spatial kNN attention (N x N) ; default: None
        """
        self.color_img = color_img
        self.patch_size = patch_size
        self.img_dim = img_dim
        self.pos_encoding = pos_encoding
        self.feat_embedding_size = feat_embedding_size
        self.knn_att_type = knn_att_type

        self.feat_extractor = feat_extractor(self.color_img,
                                             extraction_layer,
                                             hist_output_size)
        if ssl_feat_pretrain_fname is not None:
            path = r'./trained_feat_extractor'
            file = ssl_feat_pretrain_fname + '.pth'
            self.feat_extractor.load_state_dict(torch.load(os.path.join(path, file)), strict=False)
            self.feat_extractor.to("cuda:0")

        self.embedding_fc = FeatureFC(self.feat_extractor.emb_size,
                                      self.feat_embedding_size)

        if self.knn_att_type == 'representational':
            self.SkNNAtt = None
            self.RkNNAtt = kNNSelfAttention(type='representational',
                                            in_features=self.feat_embedding_size,
                                            out_features=self.feat_embedding_size,
                                            topk_R=topk_R,
                                            training=training)
        elif self.knn_att_type == 'spatial':
            self.RkNNAtt = None
            self.SkNNAtt = kNNSelfAttention(type='spatial',
                                            in_features=self.feat_embedding_size,
                                            out_features=self.feat_embedding_size,
                                            topk_S=topk_S,
                                            spatial_dist_mat=spatial_dist_mat,
                                            training=training)
        elif self.knn_att_type == 'both':
            self.RkNNAtt = kNNSelfAttention(type='representational',
                                            in_features=self.feat_embedding_size,
                                            out_features=self.feat_embedding_size,
                                            topk_R=topk_R,
                                            training=training)
            self.SkNNAtt = kNNSelfAttention(type='spatial',
                                            in_features=self.feat_embedding_size,
                                            out_features=self.feat_embedding_size,
                                            topk_S=topk_S,
                                            spatial_dist_mat=spatial_dist_mat,
                                            training=training)
        elif self.knn_att_type == 'full':
            self.RkNNAtt = None
            self.SkNNAtt = None
            self.FullAtt = kNNSelfAttention(type='full',
                                            in_features=self.feat_embedding_size,
                                            out_features=self.feat_embedding_size,
                                            training=training)
        elif self.knn_att_type == 'none':
            self.RkNNAtt = None
            self.SkNNAtt = None

        self.resblock = ResidualConnection(self.feat_embedding_size)
        
	# deepsurv
        """
        self.deepsurv_layers = nn.Sequential(
            nn.Linear(self.feat_embedding_size, att_embedding_size),
            nn.ReLU(),
            nn.Dropout(p=0.5),  # Optional dropout
            # Add more layers as required
            nn.Linear(att_embedding_size, 1)
        )
        """
        self.attention_regressor = AttentionRegressor(self.feat_embedding_size,
                                                      att_embedding_size)

    def forward(self, x):
        """
        Parameters
        ----------
        x: B x D x H x W;
            B: batch size;
            D: dimension;
                for CT-image: no. of slides;
                for X-ray image: no. of slides (=1);
                for 2D color image: no. of color channels (=3, RGB)
            H: height; W: width

        Returns
        -------
        y_prob: prediction probability
        y: prediction target (binary)
        a: attention weights
        """
        self.rknn_att = None #placeholder for rknn self-attention weights
        self.sknn_att = None #placeholder for sknn self-attention weights
        self.full_att = None #placeholder for full self-attention weights
        
        B, D, H, W = x.size()

        if self.img_dim == 3: #Adjust the number of slides (D) of the CT scans to D_ (only for 3D image)
            if D != 86:
                D_ = 86
                x = x.permute(0, 2, 1, 3)
                x = F.interpolate(x, (D_, W))
                x = x.permute(0, 2, 1, 3)

        x_patch = create_patchbag(x, self.patch_size, self.patch_size, # Construct bag of patches from image
                                  color_img=self.color_img)            # N x 1 x p1 x p2 matrix, N: Batch_size x n_patches; 1 channel; p1: patch height; p2: patch width

        h_ = self.feat_extractor(x_patch) # output representation from visual feature extractor; (Batch_size x n_patches) x Embedding_size
        h_ = ei.rearrange(h_, '(B n_patches) Emb -> B n_patches Emb', B=B) # B x n_patches x Embedding_size
        h_ = self.embedding_fc(h_) # non-linear projection of extracted image features; B x n_patches x Embedding_size

        # Positional encoding
        if self.pos_encoding:
            h_ = h_ + pos_encoding_1d(*h_.size()) # B x n_patches x Embedding_size

        # Self-attention aggregation
        if self.knn_att_type == 'representational':
            h, self.rknn_att = self.RkNNAtt(h_)
            #h = self.resblock(h_, h) # residue connection
        elif self.knn_att_type == 'spatial':
            h, self.sknn_atta = self.SkNNAtt(h_)
            #h = self.resblock(h_, h) # residue connection
        elif self.knn_att_type == 'both':
            h1, self.rknn_att = self.RkNNAtt(h_)
            h2, self.sknn_att = self.SkNNAtt(h_)
            h = (h1 + h2)/2 #Average of the representational and spatial kNN self attention
            #h = self.resblock(h_, h) # residue connection
        elif self.knn_att_type == 'full':
            h, self.full_att = self.FullAtt(h_)
            #h = self.resblock(h_, h) # residue connection
        else:
            h = h_ # No self-attention aggregation

        # deepsurv
        risk_score = self.attention_regressor(h)
        self.a = self.attention_regressor.a # Attention weights
        
        return risk_score


class kNNSelfAttention(nn.Module):
    def __init__(self, type, in_features,
                 out_features, dropout=0.3,
                 topk_R=None, topk_S=None,
                 spatial_dist_mat=None,
                 training=True):
        """
        Parameters
        ----------
        topk: int, select top-k neighbors
        type: "representational", "spatial", "both", or "full" (full self-attention)
        """
        super(kNNSelfAttention, self).__init__()
        self.type = type
        self.dropout = dropout
        self.topk_R = topk_R
        self.topk_S = topk_S
        self.spatial_dist_mat = spatial_dist_mat
        self.training = training

        self.proj = nn.Linear(in_features, out_features, bias=False)
        nn.init.kaiming_uniform_(self.proj.weight.data)

    def forward(self, x):
        """
        Parameters
        ----------
        x: feature vectors -- B x N x L
        Returns
        -------
        h: aggregated feature vector -- B x N x L
        """

        x_proj = self.proj(x) # B x N x out_features
        score = self.sim_score(x_proj, norm=False)  # B x N x N

        if self.type == 'representational':
            adj = self.RkNNAdjacency(score)
        elif self.type == 'spatial':
            adj = self.SkNNAdjacency(score)
        elif self.type == 'full':
            adj = torch.ones_like(score)

        infs = -1e19 * torch.ones_like(adj) # B x N x N
        ones = torch.ones_like(adj)
        mask = torch.where(adj == 1, ones, infs) # B x N x N
        attention_unnorm = mask * score  # B x N x N; Elementwise multiplication
        attention = F.softmax(attention_unnorm, dim=2)  # B x N x N

        h = torch.bmm(attention, x_proj) # B x N x out_features

        return h, attention

    def sim_score(self, x, norm=True):
        """
        Parameters
        ----------
        x: feature vectors -- B x N x L
        norm: if True, normalized dot product (cosine similarity), else simple dot product
        Returns
        -------
        score: similarity score matrix between each patch -- B x N x N
        """
        if norm:
            B, N, L = x.size()
            x_n = x.norm(dim=2)[:, :, None]  # Calculate norm of each vector
            x_norm = x / torch.max(x_n, 1e-8 * torch.ones_like(x_n))  # Create unit vectors
            score = torch.bmm(x_norm, torch.transpose(x_norm, 1, 2))  # calculate cosine similarity B x N x N
            # score = torch.einsum("bij, bjk -> bik", x_norm, torch.transpose(x_norm, 1, 2))
        else:
            score = torch.bmm(x, torch.transpose(x, 1, 2))  # calculate cosine similarity B x N x N

        return score

    def RkNNAdjacency(self, score):
        """
        Parameters
        ----------
        score: similarity score matrix between each patch -- B x N x N

        Returns
        -------
        adj: adjacency matrix defining k-nearest neighbors -- B x N x N
        """
        B, N, N = score.size()

        # Find kNN
        topkdim2_idx = torch.topk(score, k=self.topk_R, dim=-1)[1] # Return indice of top-k value in each row of the matrix
        topkdim2_idx = topkdim2_idx.flatten().view(B * self.topk_R * N, 1) # Reshape as (B x N x topk_R) x 1
        topkdim1_idx = torch.arange(N, device='cuda').repeat_interleave(self.topk_R).repeat(B).unsqueeze(-1) # (B x N x topk_R) x 1
        topkdim0_idx = torch.arange(B, device='cuda').repeat_interleave(N * self.topk_R).unsqueeze(-1) # (B x N x topk_R) x 1
        top_k_idx_2d =torch.cat([topkdim0_idx, topkdim1_idx, topkdim2_idx], dim=1) # (B x N x topk_R) x 3
        
        adj = torch.eye(N, device='cuda', requires_grad=False).repeat(B,1,1) # Create adjacency matrix initialized with adj[k,i,i] = 1; B x N x N
        adj[top_k_idx_2d[:,0], top_k_idx_2d[:,1], top_k_idx_2d[:,2]] = 1 # Non-symmetric adjacency matrix
        #(adj)
        return adj

    def SkNNAdjacency(self, score):
        """
        Parameters
        ----------
        score: similarity score matrix between each patch -- B x N x N

        Returns
        -------
        adj: adjacency matrix defining k-nearest neighbors -- B x N x N
        """
        B, N, N = score.size()
        
        # Spatial kNN
        zeros = torch.zeros(self.spatial_dist_mat.size())
        ones = torch.ones(self.spatial_dist_mat.size())
        # create adj matrix by replacing values larger than topk_S with 0
        adj = torch.where(self.spatial_dist_mat <= self.topk_S, ones, zeros)
        adj = adj.repeat(B,1,1).requires_grad_(False) # B x N x N
        #print('Adjacency of full self-attention', torch.unique(adj)) # Checking step

        return adj.to('cuda')


class AttentionRegressor(nn.Module):
    def __init__(self,
                 feat_embedding_size,
                 att_embedding_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feat_embedding_size, att_embedding_size),
            nn.Tanh(),
            nn.Linear(att_embedding_size, 1)
        )
        nn.init.kaiming_uniform_(self.attention[0].weight.data)
        nn.init.kaiming_uniform_(self.attention[2].weight.data)

        self.fc = nn.Sequential(
            nn.Linear(feat_embedding_size, feat_embedding_size),
            nn.ReLU(),
            nn.InstanceNorm1d(feat_embedding_size),
            nn.Dropout(0.3),
            nn.Linear(feat_embedding_size, feat_embedding_size),
            nn.ReLU(),
            nn.InstanceNorm1d(feat_embedding_size),
            nn.Dropout(0.3),
            nn.Linear(feat_embedding_size, 1)
            # nn.Sigmoid()
        )
        nn.init.kaiming_uniform_(self.fc[0].weight.data)

    def forward(self, h):
        a = self.attention(h)  # B x N x 1
        a = torch.transpose(a, 1, 2)  # B x 1 x N
        self.a = F.softmax(a, dim=2)  # B x 1 x N

        m = torch.bmm(self.a, h)  # B x 1 x L
        m = m.squeeze(1) # B x L

        y_prob = self.fc(m)
        #y_prob = F.sigmoid(y_prob)
        y_prob = torch.flatten(y_prob).float()

        return y_prob.cuda()

