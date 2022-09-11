import torch
from torch import nn
from torch.nn import functional as F

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, ksize):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, stride=1, padding=ksize//2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv2d(x)))

class ResBlock(nn.Module):
    def __init__(self, depth, in_channels, out_channels, ksize):
        super().__init__()
        self.depth = depth
        for i in range(self.depth):
            block = ConvBNReLU(in_channels if i==0 else out_channels, out_channels, ksize)
            setattr(self, f'block_{i}', block)
    
    def forward(self, x):
        for i in range(self.depth):
            block = getattr(self, f'block_{i}')
            x = x + block(x)
        return x

class UNet(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.expand_dims = nn.Conv2d(opt.color_channels, opt.dims[0], kernel_size=1)
        for i in range(opt.level):
            block1 = ConvBNReLU(opt.dims[i] if i==0 else opt.dims[i-1], opt.dims[i], opt.ksizes[i])
            block2 = ConvBNReLU(opt.dims[i], opt.dims[i], opt.ksizes[i])
            pool = nn.AvgPool2d(kernel_size=2)
            setattr(self, f'enc_block1_{i}', block1)
            setattr(self, f'enc_block2_{i}', block2)
            setattr(self, f'enc_pool_{i}', pool)
            
        for i in range(2):
            block1 = ConvBNReLU(opt.dims[-1], opt.dims[-1], opt.ksizes[-1])
            block2 = ConvBNReLU(opt.dims[-1], opt.dims[-1], opt.ksizes[-1])
            setattr(self, f'bottom_block1_{i}', block1)
            setattr(self, f'bottom_block2_{i}', block2)

        for i in range(opt.level):
            idx1 = opt.depth-1-i
            idx2 = opt.depth-2-i if i!=opt.depth-1 else 0
            block1 = ConvBNReLU(2*opt.dims[idx1], opt.dims[idx2], opt.ksizes[idx2])
            block2 = ConvBNReLU(opt.dims[idx2], opt.dims[idx2], opt.ksizes[idx2])
            setattr(self, f'dec_block1_{i}', block1)
            setattr(self, f'dec_block2_{i}', block2)
        
        self.to_out = nn.Conv2d(opt.dims[0], opt.color_channels, opt.ksizes[0], stride=1, padding=opt.ksizes[0]//2)
    
    def forward(self, x):
        x = self.expand_dims(x)

        encoded = []
        for i in range(5):
            block1 = getattr(self, f'enc_block1_{i}')
            block2 = getattr(self, f'enc_block2_{i}')
            pool = getattr(self, f'enc_pool_{i}')
            x = block1(x)
            x = block2(x)
            encoded.append(x)
            x = pool(x)
        
        for i in range(2):
            block1 = getattr(self, f'bottom_block1_{i}')
            block2 = getattr(self, f'bottom_block2_{i}')
            x = block1(x)
            x = block2(x)
        
        for i in range(5):
            block1 = getattr(self, f'dec_block1_{i}')
            block2 = getattr(self, f'dec_block2_{i}')
            x = F.interpolate(x, size=(encoded[4-i].size(2), encoded[4-i].size(3)), mode='bilinear', align_corners=False)
            x = torch.cat([x, encoded[4-i]], dim=1)
            x = block1(x)
            x = block2(x)
        x = self.to_out(x)

        return x