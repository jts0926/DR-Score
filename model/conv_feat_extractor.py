import torch
import torch.nn as nn
import torch.nn.functional as F

from efficientnet_pytorch import EfficientNet
from torchvision.models import resnet18
from kornia.feature import HardNet8, HyNet, TFeat, SOSNet, SIFTDescriptor
from torchvision.transforms.functional import rgb_to_grayscale
from .handcrafted_feat_extractor import *

class ConvNetExtractor(nn.Module):
    def __init__(self, output_length):
        # inherit nn.module
        super().__init__()

        # define conv layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 20, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(20, 50, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(50*5*5, output_length)
        )

    def forward(self, x):
        ## x is the input and is a torch.tensor
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc(x)
        return x


class ResNetExtractor(nn.Module):
    def __init__(self, num_input_ch=1):
        super().__init__()
        self.model = resnet18(pretrained=True)
        if num_input_ch == 1:
            self.model.conv1 = nn.Conv2d(num_input_ch, 64,
                                         kernel_size=(7, 7),
                                         stride=(2, 2),
                                         padding=(3, 3),
                                         bias=False)
    def forward(self, x):
        x = self.model(x)
        return x

class EfficientNetExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b0')
        self.model.to("cuda:0")

    def forward(self, x):
        x = self.model(x)
        return x


class feat_extractor(nn.Module):
    def __init__(self,
                 color_img,
                 extraction_layer,
                 hist_output_size):
        super().__init__()

        self.color_img = color_img
        self.extraction_layer = extraction_layer

        self.hist_extractor = HistogramLayer(num_bins=hist_output_size)

        if self.color_img:
            self.conv_extractor = EfficientNetExtractor()  # output size = 1000
            self.cnn_in_size = (224, 224)
            cnn_output_size = 1000
        else:
            #self.conv_extractor = ConvNetExtractor(output_length=625)  # output size = 625
            #cnn_output_size = 625
            self.conv_extractor = ResNetExtractor(num_input_ch=1)
            self.cnn_in_size = (224, 224)
            cnn_output_size = 1000

        if self.extraction_layer == 'both':
            self.emb_size = hist_output_size + cnn_output_size
        elif self.extraction_layer == 'conv':
            self.emb_size = cnn_output_size
        elif self.extraction_layer == 'hist':
            self.emb_size = hist_output_size

    def forward(self, x_patch):
        if self.extraction_layer == 'both':
            x_patch_gray = rgb_to_grayscale(x_patch) if self.color_img else x_patch
            emb_hist = self.hist_extractor(x_patch_gray)  # N x hist_bin_num
            x_patch = F.interpolate(x_patch, self.cnn_in_size)
            emb_conv = self.conv_extractor(x_patch) #N x conv_output_length
            emb = torch.cat((emb_hist, emb_conv), dim=1)

        elif self.extraction_layer == 'conv':
            x_patch = F.interpolate(x_patch, self.cnn_in_size)
            emb = self.conv_extractor(x_patch) #N x conv_output_length

        elif self.extraction_layer == 'hist':
            x_patch_gray = rgb_to_grayscale(x_patch) if self.color_img else x_patch
            #x_patch_gray = F.interpolate(x_patch_gray, (32, 32))
            emb = self.hist_extractor(x_patch_gray) #N x conv_output_length

        return emb
