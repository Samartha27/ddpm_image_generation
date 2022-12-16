import numpy as np
import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
from utils import constants
import h5py



class H5Dataset(torch.utils.data.Dataset):
    def __init__(self, h5_file, transform=None):
        self.transform = transform
        self.h5_file = h5py.File(h5_file, 'r')
        self.images = self.h5_file['images'][:]
        
    def __len__(self):
        return self.images.shape[0]
      
    def __getitem__(self, idx):
        data = self.images[idx]
        
        if self.transform:
            data = self.transform(data)
        return data

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Lambda(lambda t: (t * 2) - 1)
])

def reverse_transform(data):
    return transforms.Compose([
     Lambda(lambda t: (t + 1) / 2),
     Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
     Lambda(lambda t: t * 255.),
     Lambda(lambda t: t.numpy().astype(np.uint8)),
     ToPILImage(),
])


# image size is 64, nchw format
def get_loader(data_dir, batch_size=128, shuffle=True):
    data_train = H5Dataset(data_dir, transform = transform)
    return torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=shuffle)

# TODO: need to strip labels
def get_loader_from_pytorch(download_dir, batch_size=128, shuffle=True):
   data_train = datasets.CIFAR10(root=download_dir, train=True, download=True, transform=transform)
   return torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=shuffle)