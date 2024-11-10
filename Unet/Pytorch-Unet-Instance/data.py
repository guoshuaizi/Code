import os
import numpy as np
import torch
from torch.utils.data import Dataset
from utils import *
from torchvision import transforms


transform = transforms.Compose([
    transforms.ToTensor()
])


class MyDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.name = os.listdir(os.path.join(path, 'Images'))

    def __len__(self):
        self.data_len= len(self.name)
        return len(self.name)

    def __getitem__(self, index):
        segment_name = self.name[index % self.data_len]  # xx.png
        segment_path = os.path.join(self.path, 'Labels', segment_name)
        image_path = os.path.join(self.path, 'Images', segment_name)
        segment_image = keep_image_size_open(segment_path)
        image = keep_image_size_open_rgb(image_path)
        return transform(preprocess_input(np.array(image,np.float32))), torch.Tensor(preprocess_input(np.array(segment_image,np.float32)))



