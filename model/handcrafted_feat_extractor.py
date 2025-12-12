import torch
import torch.nn as nn
import numpy as np


class HistogramLayer(nn.Module):
    def __init__(self, kernel_size=2, num_bins=10,
                 stride=1, padding=0, normalize_count=True,
                 normalize_bins=True, global_pool=True,
                 count_include_pad=False, ceil_mode=False):

        # inherit nn.module
        super(HistogramLayer, self).__init__()

        # define layer properties
        # histogram bin data
        self.numBins = num_bins
        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = padding
        self.normalize_count = normalize_count
        self.normalize_bins = normalize_bins
        self.global_pool = global_pool
        self.count_include_pad = count_include_pad
        self.ceil_mode = ceil_mode

        # For each data type, apply two 1x1 convolutions, 1) to learn bin center (bias)
        # and 2) to learn bin width

        self.bin_centers_conv = nn.Conv2d(1, self.numBins, 1,
                                          groups=1, bias=True)
        self.bin_centers_conv.weight.data.fill_(1)
        self.bin_centers_conv.weight.requires_grad = False
        self.bin_widths_conv = nn.Conv2d(self.numBins, self.numBins, 1,
                                         groups=self.numBins, bias=False)
        if global_pool == True:
            self.hist_pool = nn.AdaptiveAvgPool2d(1)
        else:
            self.hist_pool = nn.AvgPool2d(self.kernel_size, stride=self.stride,
                                          padding=self.padding, ceil_mode=self.ceil_mode,
                                          count_include_pad=self.count_include_pad)
        self.centers = self.bin_centers_conv.bias
        self.widths = self.bin_widths_conv.weight

    def forward(self, xx):
        ## xx is the input and is a torch.tensor
        ##each element of output is the frequency for the bin for that window

        # Pass through first convolution to learn bin centers
        xx = self.bin_centers_conv(xx)

        # Pass through second convolution to learn bin widths
        xx = self.bin_widths_conv(xx)

        # Pass through radial basis function
        xx = torch.exp(-(xx ** 2))

        # Enforce sum to one constraint
        # Add small positive constant in case sum is zero
        if (self.normalize_bins):
            xx = self.constrain_bins(xx)

        # Get localized histogram output, if normalize, average count
        if (self.normalize_count):
            xx = self.hist_pool(xx)
            xx = torch.flatten(xx, start_dim=1, end_dim=-1)
        else:
            xx = torch.Tensor(np.prod(np.asarray(self.hist_pool.kernel_size)) * self.hist_pool(xx))

        return xx

    def constrain_bins(self, xx):
        # Enforce sum to one constraint across bins
        n, c, h, w = xx.size()
        xx_sum = xx.reshape(n, c // self.numBins, self.numBins, h, w).sum(2) + torch.tensor(10e-6)
        xx_sum = torch.repeat_interleave(xx_sum, self.numBins, dim=1)
        xx = xx / xx_sum

        return xx

class FourierHistogramLayer(nn.Module):
    def __init__(self, kernel_size=2, num_bins=4,
                 stride=1, padding=0, normalize_count=True,
                 normalize_bins=True, global_pool=True,
                 count_include_pad=False, ceil_mode=False):

        # inherit nn.module
        # Fourier-transformed histogram module to extract spectrum of fourier features
        # Fourier series coefficients are extracted
        super(FourierHistogramLayer, self).__init__()

        # define layer properties
        # histogram bin data
        self.numBins = num_bins
        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = padding
        self.normalize_count = normalize_count
        self.normalize_bins = normalize_bins
        self.global_pool = global_pool
        self.count_include_pad = count_include_pad
        self.ceil_mode = ceil_mode

        # For each data type, apply two 1x1 convolutions, 1) to learn bin center (bias)
        # and 2) to learn bin width

        self.bin_centers_conv = nn.Conv2d(1, self.numBins, 1,
                                          groups=1, bias=True)
        self.bin_centers_conv.weight.data.fill_(1)
        self.bin_centers_conv.weight.requires_grad = False
        self.bin_widths_conv = nn.Conv2d(self.numBins, self.numBins, 1,
                                         groups=self.numBins, bias=False)
        if global_pool == True:
            self.hist_pool = nn.AdaptiveAvgPool2d(1)
        else:
            self.hist_pool = nn.AvgPool2d(self.kernel_size, stride=self.stride,
                                          padding=self.padding, ceil_mode=self.ceil_mode,
                                          count_include_pad=self.count_include_pad)
        self.centers = self.bin_centers_conv.bias
        self.widths = self.bin_widths_conv.weight

    def forward(self, xx):
        ## xx is the input and is a torch.tensor
        ##each element of output is the frequency for the bin for that window

        # Pass through first convolution to learn bin centers
        xx = self.bin_centers_conv(xx)

        # Pass through second convolution to learn bin widths
        xx = self.bin_widths_conv(xx)

        # Pass through radial basis function
        xx = torch.exp(-(xx ** 2))

        # Enforce sum to one constraint
        # Add small positive constant in case sum is zero
        if (self.normalize_bins):
            xx = self.constrain_bins(xx)

        # Get localized histogram output, if normalize, average count
        if (self.normalize_count):
            xx = self.hist_pool(xx)
        else:
            xx = torch.Tensor(np.prod(np.asarray(self.hist_pool.kernel_size)) * self.hist_pool(xx))

        return xx

    def constrain_bins(self, xx):
        # Enforce sum to one constraint across bins
        n, c, h, w = xx.size()
        xx_sum = xx.reshape(n, c // self.numBins, self.numBins, h, w).sum(2) + torch.tensor(10e-6)
        xx_sum = torch.repeat_interleave(xx_sum, self.numBins, dim=1)
        xx = xx / xx_sum

        return xx

class FeatureFC(nn.Module):
    def __init__(self, input_length, output_length):
        super(FeatureFC, self).__init__()
        self.input_length = input_length
        self.output_length = output_length

        self.fc = nn.Sequential(nn.LayerNorm(input_length),
                                nn.Linear(input_length, output_length),
                                nn.ReLU(inplace=True))

    def forward(self, x):
        return self.fc(x)
