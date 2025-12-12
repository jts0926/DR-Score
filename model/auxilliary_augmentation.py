import torch
import torch.nn as nn
import kornia.filters as kfilters

class SobelFilter(nn.Module):
    def __init__(self, normalized=True):
        super().__init__()
        self.normalized = normalized

    def forward(self, x):
        return kfilters.sobel(x, normalized=self.normalized)

class LaplacianFilter(nn.Module):
    def __init__(self,
                 kernel_size=3,
                 border_type='reflect',
                 normalized=True):
        super().__init__()
        """
        Params:
        kernel_size (int): the size of the kernel.
        border_type (str, optional): The padding mode to be applied before convolving. 
                                     The expected modes are: 'constant', 'reflect', 'replicate' or 'circular'. Default: 'reflect'
        normalized (bool, optional): if True, L1 norm of the kernel is set to 1. Default: True
        """
        self.kernel_size = kernel_size
        self.border_type = border_type
        self.normalized = normalized

    def forward(self, x):
        return kfilters.laplacian(x, self.kernel_size,
                                  self.border_type,
                                  self.normalized)

class SpatialGradientFilter(nn.Module):
    def __init__(self,
                 mode='diff',
                 order=1,
                 normalized=True):
        super().__init__()
        """
        Params:
        mode (str, optional): derivatives modality, can be: sobel or diff. Default: 'diff'
        order (int, optional): the order of the derivatives. Default: 1
        normalized (bool, optional): whether the output is normalized. Default: True
        """
        self.mode = mode
        self.order = order
        self.normalized = normalized

    def forward(self, x):
        return kfilters.spatial_gradient(x, self.mode,
                                         self.order,
                                         self.normalized)

class UnsharpMask(nn.Module):
    def __init__(self,
                 kernel_size=(3,3),
                 sigma=(1.5, 1.5),
                 border_type='reflect'):
        super().__init__()
        """
        Params:
        kernel_size (Tuple[int, int]): the size of the kernel.
        sigma (Tuple[float, float]): the standard deviation of the kernel.
        border_type (str, optional): the padding mode to be applied before convolving. 
                                     The expected modes are: 'constant', 'reflect', 'replicate' or 'circular'. 
                                     Default: 'reflect'
        """
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.border_type = border_type

    def forward(self, x):
        return kfilters.unsharp_mask(x, self.kernel_size,
                                     self.sigma,
                                     self.border_type)