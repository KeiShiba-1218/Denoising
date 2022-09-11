import torch
from torch import nn
import torchinfo
import json
from tqdm import tqdm
from easydict import EasyDict
from PIL import Image
import glob
import cv2
import time
import timeit
import sys

#from models.unet import UNet
from models.resunet import UNet
#from models.axialunet import AxialUNet
from dataloader import DIVFlicker2KDataset, ValDataset
from utils.utils import tensor2ndarray
from metrics import calculate_psnr, calculate_ssim
from models.transformer import TransformerBlock

def test_block():
    device = torch.device('cuda')
    with open('config/config_unet_gray.json', 'r', encoding='utf-8')  as fp:
        opt = EasyDict(json.load(fp))

    x = torch.rand(size=(1,64,8,8)).to(device)

    net = TransformerBlock(64,8,1).to(device)

    out = net(x)
    print(out.shape)

    torchinfo.summary(net, input_data=[x])

def test_dataset():
    with open('config/config_unet.json', 'r', encoding='utf-8')  as fp:
        opt = EasyDict(json.load(fp))
    dataset = DIVFlicker2KDataset(opt)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)
    
    for img, gt in dataloader:
        img = tensor2ndarray(img)
        gt = tensor2ndarray(gt)

        for b in range(8):
            Image.fromarray(img[b,:,:,:]).save(f'temp/{b}_jpeg.png')
            Image.fromarray(gt[b,:,:,:]).save(f'temp/{b}_png.png')
        
        exit()

def test_val_dataset():
    with open('config/config_unet.json', 'r', encoding='utf-8')  as fp:
        opt = EasyDict(json.load(fp))
    dataset = ValDataset(opt, 10)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
    
    for img, gt in dataloader:
        img = tensor2ndarray(img)
        gt = tensor2ndarray(gt)

        for b in range(1):
            Image.fromarray(img[b,:,:,:]).save(f'temp/{b}_jpeg.png')
            Image.fromarray(gt[b,:,:,:]).save(f'temp/{b}_png.png')
        
        exit()

def eval_images():
    images1 = sorted(glob.glob('datasets/Classic5/10/*.jpg'))
    images2 = sorted(glob.glob('datasets/Classic5/raw/*.png'))

    psnr, ssim = 0, 0
    for img_path1, img_path2 in zip(images1, images2):
        img1 = cv2.imread(img_path1)
        img2 = cv2.imread(img_path2)
        psnr += calculate_psnr(img1, img2, crop_border=0, test_y_channel=False)
        ssim += calculate_ssim(img1, img2, crop_border=0, test_y_channel=False)
    psnr = psnr / len(images1)
    ssim = ssim / len(images1)
    print(psnr, ssim)

def eval_from_comparison():
    images = sorted(glob.glob('experiments/resunet2/generated/10/470000/*.png'))
    psnrj, ssimj = 0, 0
    psnr, ssim = 0, 0
    for img_path in images:
        compare_img = cv2.imread(img_path)
        h, w = compare_img.shape[0], compare_img.shape[1]//3
        img = compare_img[:, 0:w]
        gen = compare_img[:, w:2*w]
        gt = compare_img[:, 2*w:3*w]
        psnrj += calculate_psnr(img, gt, crop_border=0, test_y_channel=False)
        ssimj += calculate_ssim(img, gt, crop_border=0, test_y_channel=False)
        psnr += calculate_psnr(gen, gt, crop_border=0, test_y_channel=False)
        ssim += calculate_ssim(gen, gt, crop_border=0, test_y_channel=False)
    psnrj = psnrj / len(images)
    ssimj = ssimj / len(images)
    psnr = psnr / len(images)
    ssim = ssim / len(images)
    print(f'PSNR: {psnrj:f}, SSIM: {ssimj:f}')
    print(f'PSNR: {psnr:f}, SSIM: {ssim:f}')

def test_speed():
    device = torch.device('cuda')
    loop = 1000
    batch_size = 1
    resolution = int(sys.argv[1])
    channels = int(sys.argv[2])
    dtype = torch.float32
    x = torch.rand((batch_size, channels, resolution, resolution)).to(device, dtype)
    conv = nn.Conv2d(channels, channels, kernel_size=3).to(device, dtype)
    cconv = nn.Conv2d(channels, channels, kernel_size=1).to(device, dtype)
    dconv = nn.Conv2d(channels, channels, kernel_size=3, groups=channels).to(device, dtype)

    torch.backends.cudnn.benchmark = True

    def conv_forward():
        torch.cuda.synchronize(device)
        out = conv(x)
        torch.cuda.synchronize(device)
    
    def dsconv_forward():
        torch.cuda.synchronize(device)
        out = cconv(x)
        out = dconv(x)
        torch.cuda.synchronize(device)
    
    with torch.no_grad():
        res = timeit.timeit(lambda: conv_forward(), number=loop)
    print(f'{resolution},{channels},{res/loop*1000:f}')
    

if __name__=='__main__':
    test_block()