import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import monai as mi
import pytorch_lightning as pl
#from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import kornia.augmentation as ktransforms

from .conv_feat_extractor import *
from .handcrafted_feat_extractor import *
from .auxilliary_layer import *
from .auxilliary_augmentation import *
from .ssl_loss import *

class augmentation:
    def __init__(self, img_size):
        blur = transforms.GaussianBlur((3, 3), (0.1, 2.0))
        sobel = SobelFilter()
        laplacian = LaplacianFilter()
        gradient = SpatialGradientFilter()
        unsharp = UnsharpMask()
        box_blur = ktransforms.RandomBoxBlur(p=0.5)
        equalize = ktransforms.RandomEqualize(p=0.5)
        g_noise = ktransforms.RandomGaussianNoise(mean=0.0, std=1.0, p=0.5)
        posterize = ktransforms.RandomPosterize(p=0.5)
        sharpness = ktransforms.RandomSharpness(sharpness=0.5, same_on_batch=False, p=0.5)

        color_jitter = ktransforms.RandomPlanckianJitter(mode='blackbody', p=0.5)
        contrast = ktransforms.RandomPlasmaContrast(roughness=(0.1, 0.7), p=0.5)
        color_jiggle = ktransforms.ColorJiggle(p=0.5)
        channel_shuffle = ktransforms.RandomChannelShuffle(p=0.5)
        rgb_shift = ktransforms.RandomRGBShift(r_shift_limit=0.5, g_shift_limit=0.5, b_shift_limit=0.5, p=0.5)

        self.compose_trsf = nn.Sequential(
                                transforms.RandomApply([blur,
                                                        sobel,
                                                        laplacian,
                                                        unsharp], p=0.5),
                                transforms.RandomHorizontalFlip(p=0.5),
                                #box_blur,
                                #g_noise,
                                #posterize,
                                sharpness,
                                #transforms.Normalize(0.485, 0.229),
                                transforms.RandomResizedCrop(size=img_size)
                                )

    def __call__(self, x):
        return self.compose_trsf(x),  self.compose_trsf(x)

class proj_layer(nn.Module):
    def __init__(self,
                 cnn_emb_size,
                 out_emb_size):
        super().__init__()
        self.proj = nn.Sequential(
                        nn.Linear(cnn_emb_size, cnn_emb_size),
                        nn.BatchNorm1d(cnn_emb_size),
                        nn.ReLU(),
                        nn.Linear(cnn_emb_size, out_emb_size),
                        nn.BatchNorm1d(out_emb_size)
                        )

    def forward(self, x_emb):
        x = self.proj(x_emb)
        return x

class SimCLR(pl.LightningModule):
    def __init__(self,
                 img_dim,
                 patch_size,
                 color_img,
                 batch_size,
                 extraction_layer,
                 hist_output_size,
                 temperature,
                 lr,
                 weight_decay):
        super().__init__()

        self.img_dim = img_dim
        self.color_img = color_img
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.temperature = temperature
        self.lr = lr
        self.weight_decay = weight_decay

        self.feat_extractor = feat_extractor(self.color_img,
                                             extraction_layer,
                                             hist_output_size)

        self.proj = proj_layer(cnn_emb_size=self.feat_extractor.emb_size,
                               out_emb_size=self.feat_extractor.emb_size//4)

        self.model = nn.Sequential(self.feat_extractor,
                                   self.proj)

        self.augment = augmentation(img_size=self.feat_extractor.cnn_in_size)

        self.loss = NT_XentLoss(self.temperature, self.batch_size)

    def forward(self, X):
        x_patch = self._create_patch_set(X)
        emb = self.model(x_patch)
        return emb

    def _create_patch_set(self, x):
        if self.img_dim == 3:  #Adjust the number of slides (D) of the CT scans to D_ (only for 3D image)
            B, D, H, W = x.size()
            if D != 86:
                D_ = 86
                x = x.permute(0, 2, 1, 3)
                x = F.interpolate(x, (D_, W))
                x = x.permute(0, 2, 1, 3)

        x_patch = create_patchbag(x, self.patch_size, self.patch_size,  #Construct bag of patches from image
                                  color_img=self.color_img)  #N x 1 x p1 x p2 matrix, N: # of patches; 1 channel; p1: patch height; p2: patch width
        b, c, h, w = x_patch.size()

        samp_idx = torch.tensor(np.random.choice(b,
                                                 self.batch_size,
                                                 replace=False),
                                dtype=torch.long) #Sample batch_size of patches from the patch bag without replacement

        return x_patch[samp_idx, :, :, :]

    def training_step(self, batch, batch_idx):
        x, label = batch
        x_patch = self._create_patch_set(x) #sample patches to form a batch from the image
        x1, x2 = self.augment(x_patch)
        emb1 = self.model(x1)
        emb2 = self.model(x2)
        loss = self.loss(emb1, emb2)
        self.log('Contrastive loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                                       0.999,
                                                                       last_epoch=- 1,
                                                                       verbose=False)
        return [optimizer], [scheduler]

    def save(self, fname):
        path = r'./trained_feat_extractor'
        file = fname + '.pth'
        torch.save(self.model.state_dict(), os.path.join(path, file))

    def load(self, fname):
        path = r'./trained_feat_extractor'
        file = fname + '.pth'
        self.model.load_state_dict(torch.load(os.path.join(path, file)), strict=False)
        self.model.to("cuda:0")
        return self



