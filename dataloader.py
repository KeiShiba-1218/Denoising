import torch
import torch.utils.data
from torchvision import transforms
from torchvision.transforms import functional as TF
import torchvision
import numpy as np
import json
import glob
import os
from PIL import Image
import random
import io
import numpy as np

def random_jpeg_compression(img):
    # qf = random.randint(8, 96) if random.random() > 0.75 else random.choice([10,20,30,40,50,60])
    qf = int(np.clip(np.abs(np.random.normal(0,30)+10), 10, 95))

    output_io_stream = io.BytesIO()
    img.save(output_io_stream, 'JPEG', quality=qf, optimice=True)
    output_io_stream.seek(0)
    return Image.open(output_io_stream)

class DIVFlicker2KDataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.images = sorted(glob.glob(os.path.join(opt.dataset_path, '*.png')))
        self.transform_input = transforms.Compose([
            transforms.Lambda(random_jpeg_compression),
            transforms.ToTensor(),
        ])
        self.transform_gt = transforms.Compose([
            transforms.ToTensor(),
        ])
        
    def __getitem__(self, idx):
        img_path = self.images[idx]
        with Image.open(img_path) as img_base:
            img = self.transform_input(img_base)
            gt = self.transform_gt(img_base)
        return img, gt
    
    def __len__(self):
        return len(self.images)

class DIVFlicker2KDatasetCrop(torch.utils.data.Dataset):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.images = sorted(glob.glob(os.path.join(opt.dataset_path, '*.png')))

        self.transform_base = transforms.Compose([
            transforms.RandomCrop(opt.input_image_resolution), 
        ])
        self.transform_input = transforms.Compose([
            transforms.Lambda(random_jpeg_compression),
            transforms.ToTensor(),
        ])
        self.transform_gt = transforms.Compose([
            transforms.ToTensor(),
        ])
        
    def __getitem__(self, idx):
        img_path = self.images[idx]
        with Image.open(img_path) as img_base:
            img_base = self.transform_base(img_base)
            img = self.transform_input(img_base)
            gt = self.transform_gt(img_base)
            
        return img, gt
    
    def __len__(self):
        return len(self.images)

class ValDataset(torch.utils.data.Dataset):
    def __init__(self, opt, qf):
        super().__init__()
        self.opt = opt
        self.noise_images = sorted(glob.glob(os.path.join(opt.val_dataset_path, str(qf), '*.jpg')))
        self.gt_images = sorted(glob.glob(os.path.join(opt.val_dataset_path, 'raw', '*.png')))
    
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    
    def __getitem__(self, idx):
        img_path = self.noise_images[idx]
        gt_path = self.gt_images[idx]
        with Image.open(img_path) as img, Image.open(gt_path) as gt:
            img = self.transform(img)
            gt = self.transform(gt)

            return img, gt
    
    def __len__(self):
        return len(self.noise_images)