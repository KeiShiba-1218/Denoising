import os
import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.data
from PIL import Image
from tqdm import tqdm
import argparse
from collections import OrderedDict
import glob
import cv2

from utils.utils import tensor2ndarray, load_option
from dataloader import DIVFlicker2KDataset, ValDataset
from easydict import EasyDict
from models.resunet import UNet
from metrics import calculate_psnr, calculate_ssim

def generate_images(opt, checkpoint_path, out_dir):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    dataset_paths = ['datasets/Classic5', 'datasets/LIVE1_gray'] if opt.color_channels==1 else ['datasets/LIVE1_color']
    for dataset_path in dataset_paths:
        dataset_name = os.path.basename(dataset_path)
        print(f'Processing {dataset_name}')
        out_dataset_dir = os.path.join(out_dir, dataset_name)
        os.makedirs(out_dataset_dir, exist_ok=True)
        os.makedirs(os.path.join(out_dataset_dir, 'GT'), exist_ok=True)

        net = UNet(opt).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        net.load_state_dict(checkpoint['netG_state_dict'])
        net.eval()

        for QF in tqdm([10,20,30,40,50,60,70,80,90]):
            os.makedirs(os.path.join(out_dataset_dir, 'generated', str(QF)), exist_ok=True)
            os.makedirs(os.path.join(out_dataset_dir, 'comparison', str(QF)), exist_ok=True)

            opt.val_dataset_path = dataset_path
            val_dataset = ValDataset(opt, QF)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)
            for i, (img, gt) in enumerate(val_loader):
                img, gt = img.to(device), gt.to(device)
                with torch.no_grad():
                    out = net(img)
                    out = out.sigmoid()
                
                img = tensor2ndarray(img)
                gen = tensor2ndarray(out)
                gt = tensor2ndarray(gt)

                if opt.color_channels==1:
                    img = Image.fromarray(img[0,:,:,0])
                    gen = Image.fromarray(gen[0,:,:,0])
                    gt = Image.fromarray(gt[0,:,:,0])
                else:
                    img = Image.fromarray(img[0,:,:,:])
                    gen = Image.fromarray(gen[0,:,:,:])
                    gt = Image.fromarray(gt[0,:,:,:])

                fname = f'{i:03}.png'
                if i==0: gt.save(os.path.join(out_dataset_dir, 'GT', fname), 'PNG')
                gen.save(os.path.join(out_dataset_dir, 'generated', str(QF), fname), 'PNG')
                
                compare_img = Image.new('RGB' if opt.color_channels==3 else 'L', size=(3*img.width, img.height), color=0)
                compare_img.paste(img, box=(0, 0))
                compare_img.paste(gen, box=(img.width, 0))
                compare_img.paste(gt, box=(2*img.width, 0))
                compare_img.save(os.path.join(out_dataset_dir, 'comparison', str(QF), fname), 'PNG')

def eval_from_path(gen_path, gt_path):
    images1 = sorted(glob.glob(os.path.join(gen_path, '*.png')))
    images2 = sorted(glob.glob(os.path.join(gt_path, '*.png')))
    psnr, ssim = 0, 0
    for img_path1, img_path2 in zip(images1, images2):
        img1 = cv2.imread(img_path1)
        img2 = cv2.imread(img_path2)
        psnr += calculate_psnr(img1, img2, crop_border=0, test_y_channel=False)
        ssim += calculate_ssim(img1, img2, crop_border=0, test_y_channel=False)
    psnr = psnr / len(images1)
    ssim = ssim / len(images1)
    return psnr, ssim

def evaluation(opt, out_dir):
    dataset_paths = ['datasets/Classic5', 'datasets/LIVE1_gray'] if opt.color_channels==1 else ['datasets/LIVE1_color']
    for dataset_path in dataset_paths:
        dataset_name = os.path.basename(dataset_path)
        print(f'Processing {dataset_name}')
        out_dataset_dir = os.path.join(out_dir, dataset_name)
        with open(os.path.join(out_dataset_dir, 'results.csv'), 'w', encoding='utf-8') as fp:
            fp.write('dataset,qf,psnr,ssim\n')

        gt_path = os.path.join(out_dataset_dir, 'GT')

        for QF in tqdm([10,20,30,40,50,60,70,80,90]):
            gen_path = os.path.join(out_dataset_dir, 'generated', str(QF))
            psnr, ssim = eval_from_path(gen_path, gt_path)
            print(f'{dataset_name}: QF: {QF}, PSNR: {psnr:f}, SSIM: {ssim:f}')
            with open(os.path.join(out_dataset_dir, 'results.csv'), 'a', encoding='utf-8') as fp:
                fp.write(f'{dataset_name},{QF},{psnr},{ssim}\n')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='A script of generate images.')
    parser.add_argument('-c', '--config', required=True, help='Path of config file')
    parser.add_argument('-ckpt', '--checkpoint_path', default=None, help='Path to the chenckpoint')
    args = parser.parse_args()
    opt = EasyDict(load_option(args.config))
        
    model_name = opt.name
    if args.checkpoint_path==None:
        checkpoint_path = os.path.join('experiments', model_name, 'ckpt', f'{model_name}_{opt.steps}.ckpt')
    else:
        checkpoint_path = args.checkpoint_path
    
    out_dir = f'results/{model_name}'
    
    generate_images(opt, checkpoint_path, out_dir)