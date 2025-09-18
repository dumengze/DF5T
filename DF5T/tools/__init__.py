import os
import torch
import numbers
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torch.utils.data import Subset
import numpy as np
import torchvision
from PIL import Image
from functools import partial

class Crop(object):
    def __init__(self, x1, x2, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def __call__(self, img):
        return F.crop(img, self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1)

    def __repr__(self):
        return self.__class__.__name__ + "(x1={}, x2={}, y1={}, y2={})".format(
            self.x1, self.x2, self.y1, self.y2
        )

def center_crop_arr(pil_image, image_size=256):
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    crop_y = max(0, crop_y)
    crop_x = max(0, crop_x)
    crop_height = min(image_size, arr.shape[0] - crop_y)
    crop_width = min(image_size, arr.shape[1] - crop_x)
    return arr[crop_y : crop_y + crop_height, crop_x : crop_x + crop_width]

def get_dataset(args, config):
    if config.data.random_flip is False:
        tran_transform = test_transform = transforms.Compose(
            [transforms.ToTensor()]  
        )
    else:
        tran_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5), 
                transforms.ToTensor(),
            ]
        )
        test_transform = transforms.Compose(
            [transforms.ToTensor()] 
        )

    if  config.data.dataset == 'MitEM':
        if config.data.subset_1k:
            from tools.data import ImageDataset
            dataset = ImageDataset(os.path.join(args.exp, 'datasets', 'MitEM', 'MitEM'),
                     os.path.join(args.exp, 'MitEM_val_1k.txt'),
                     normalize=False)
            test_dataset = dataset
        elif config.data.out_of_dist:
            dataset = torchvision.datasets.ImageFolder(
                os.path.join(args.exp, 'datasets', 'ood'),
                transform=transforms.Compose([
                    transforms.ToTensor() 
                ])
            )
            test_dataset = dataset
        else:
            dataset = torchvision.datasets.ImageNet(
                os.path.join(args.exp, 'datasets', 'MitEM'), split='val',
                transform=transforms.Compose([
                    transforms.ToTensor()
                ])
            )
            test_dataset = dataset
    else:
        dataset, test_dataset = None, None

    return dataset, test_dataset

def logit_transform(image, lam=1e-6):
    image = lam + (1 - 2 * lam) * image
    return torch.log(image) - torch.log1p(-image)


def data_transform(config, X):
    if config.data.uniform_dequantization:
        X = X / 256.0 * 255.0 + torch.rand_like(X) / 256.0
    if config.data.gaussian_dequantization:
        X = X + torch.randn_like(X) * 0.01

    if config.data.rescaled:
        X = 2 * X - 1.0
    elif config.data.logit_transform:
        X = logit_transform(X)

    if hasattr(config, "image_mean"):
        return X - config.image_mean.to(X.device)[None, ...]

    return X


def inverse_data_transform(config, X):
    if hasattr(config, "image_mean"):
        X = X + config.image_mean.to(X.device)[None, ...]

    if config.data.logit_transform:
        X = torch.sigmoid(X)
    elif config.data.rescaled:
        X = (X + 1.0) / 2.0

    return torch.clamp(X, 0.0, 1.0)