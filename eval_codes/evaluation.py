import glob
import os
import argparse

import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import torchvision.transforms.functional
import torch.utils.data
from PIL import Image
from tqdm import tqdm
import numpy as np
from easydict import EasyDict

from dataloader import ValDataset
from metrics import calculate_psnr, calculate_ssim
from utils.utils import load_option



def eval_from_image(out_path, gen_path, gt_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    images_gen = sorted(glob.glob(os.path.join(gen_path, '*.png')))
    images_gt = sorted(glob.glob(os.path.join(gt_path, '*.png')))
    n_images = len(images_gen)
    
    psnr, ssim = 0.0, 0.0
    with torch.no_grad():
        for imgpath_gen, imgpath_gt in zip(tqdm(images_gen), images_gt):
            with Image.open(imgpath_gen) as img_gen, Image.open(imgpath_gt) as img_gt:
                img_gen = np.array(img_gen)
                img_gt = np.array(img_gt)
                psnr += calculate_psnr(img_gen, img_gt, crop_border=0, test_y_channel=True) / n_images
                ssim += calculate_ssim(img_gen, img_gt, crop_border=0, test_y_channel=True) / n_images
                img_gen = (torch.tensor(img_gen).permute(2,0,1) / 255.0).to(device)
                img_gt = (torch.tensor(img_gt).permute(2,0,1) / 255.0).to(device)
    
        dataset_gen = SimpleImageDataset(gen_path)
        

    print(f'PSNR: {psnr:f}, SSIM: {ssim:f}')
    
    with open(os.path.join(out_path, 'results.txt'), 'w', encoding='utf-8') as fp:
        fp.write(f'PSNR: {psnr:f}, SSIM: {ssim:f}')
    

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='A script of evaluate metrics.')
    parser.add_argument('-c', '--config', required=True, help='Path of config file')
    args = parser.parse_args()
    opt = EasyDict(load_option(args.config))
    model_name = opt.name
    
    out_path = f'results/{model_name}'
    gen_path = f'results/{model_name}/generated'
    gt_path = f'results/{model_name}/GT'
    

    eval_from_image(out_path, gen_path, gt_path)