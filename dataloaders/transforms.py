import torch
from torchvision import transforms
import monai as mi
from PIL import Image
import PIL
from skimage import exposure
import cv2
import numpy as np

def GetTrainTransforms(model_input_size=(224,224)):
    return transforms.Compose([transforms.ToTensor(),
                               transforms.Resize(model_input_size),
                               transforms.RandomPerspective(distortion_scale=0.2, p=0.1),
                               transforms.RandomHorizontalFlip(p=0.5),
                               #transforms.RandomRotation(degrees=(0,10)),
                               #transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                               #transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
                               mi.transforms.RandZoom(prob=0.2, max_zoom=1.5),
                               mi.transforms.RandGaussianNoise(prob=0.2),
                               mi.transforms.RandStdShiftIntensity(0.5, prob=0.2),
                               mi.transforms.RandAdjustContrast(prob=0.2),
                               mi.transforms.RandGaussianSmooth(prob=0.2),
                               mi.transforms.RandHistogramShift(prob=0.2),
                               mi.transforms.RandKSpaceSpikeNoise(prob=0.2),
                               mi.transforms.NormalizeIntensity(),
                               transforms.Resize(model_input_size)])

def GetValidTransforms(model_input_size=(224,224)):
    return transforms.Compose([transforms.ToTensor(),
                               mi.transforms.NormalizeIntensity(),
                               transforms.Resize(model_input_size)])

def GetBasicTransforms(model_input_size=(224,224)):
    return transforms.Compose([transforms.ToTensor(),
                               mi.transforms.NormalizeIntensity(),
                               transforms.Resize(model_input_size)])
class CLAHE_transform(object):
    def __init__(self, output_format='tensor'):
        self.output_format = output_format
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def __call__(self, img):
        """
        :param img: PIL.Image, np.array or torch.Tensor
        :return: transformed image, torch.Tensor
        """
        if isinstance(img, PIL.Image.Image):
            img = np.asarray(img)
            if img.ndim > 2:
                img = np.squeeze(img, axis=0)
        elif isinstance(img, torch.Tensor):
            img = img.squeeze(0)
            img = img.detach().cpu().numpy()
            print(img)
        elif isinstance(img, np.ndarray):
            pass

        cl_img = self.clahe.apply(img)

        if self.output_format == 'tensor':
            cl_img = cl_img.unsquuze(0)
            cl_img = torch.from_numpy(cl_img)
        elif self.output_format == 'pil':
            cl_img = Image.fromarray(cl_img)
        return cl_img

    def __repr__(self):
        return self.__class__.__name__+'()'