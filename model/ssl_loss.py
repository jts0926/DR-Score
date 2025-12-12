import torch
import torch.nn as nn
import torch.nn.functional as F

class NT_XentLoss(nn.Module):
    def __init__(self,
                 temperature,
                 batch_size):
        """
        Parameters
        ----------
        temperature: temperature parameter
        batch_size
        """
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature))
        self.register_buffer("negatives_mask", (~torch.eye(self.batch_size * 2, self.batch_size * 2, dtype=bool)).float())

        self.cos_sim = nn.CosineSimilarity(dim=2)

    def forward(self, emb_i, emb_j):
        """
        Parameters
        ----------
        emb_i: transformed embedding 1
        emb_j: transformed embedding 2
        """
        self.batch_size, _ = emb_i.size()
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        rep = torch.cat([z_i, z_j], dim=0)
        rep_n = rep.norm(dim=1)[:, None] # Calculate norm of each vector
        rep_norm = rep / torch.max(rep_n, 1e-8 * torch.ones_like(rep_n))
        sim_mat = torch.mm(rep_norm, rep_norm.transpose(0,1)) # Calculate cos similarity matrix

        sim_ij = torch.diag(sim_mat, self.batch_size) # similarity between i-th and (bs + 1)-th patches (same patch but different trans
        sim_ji = torch.diag(sim_mat, -self.batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives / self.temperature)
        denominator = self.negatives_mask * torch.exp(sim_mat / self.temperature)

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = (1 / (2 * self.batch_size)) * torch.sum(loss_partial)

        return loss