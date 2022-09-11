import torch
from torch import nn
from torch.nn import functional as F
from models.positional_encoding import ConditionalPositionalEncoding
from models.axial_transformer import AxialTransformerBlock, AxialTransformerDecoderBlock

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, ksize):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, stride=1, padding=ksize//2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv2d(x)))

class ResBlock(nn.Module):
    def __init__(self, channels, ksize, mode='BRC'):
        super().__init__()
        layer_dict = {
            'C': nn.Conv2d(channels, channels, ksize, padding=ksize//2), 
            'R': nn.ReLU(inplace=True),
            'B': nn.BatchNorm2d(channels),
        }
        layers = []
        for m in mode:
            layers.append(layer_dict[m])
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return x + self.layers(x)

class ResBlocks(nn.Module):
    def __init__(self, depth, in_channels, out_channels, ksize):
        super().__init__()
        self.depth = depth

        self.expand_dims = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        for i in range(self.depth):
            block = ResBlock(out_channels, ksize, mode='BRC')
            setattr(self, f'block_{i}', block)
    
    def forward(self, x):
        x = self.expand_dims(x)
        for i in range(self.depth):
            block = getattr(self, f'block_{i}')
            x = block(x)
        return x

class UNetEncoderSpatial(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.expand_dims = nn.Conv2d(opt.color_channels, opt.dims[0], kernel_size=1)
        for i in range(opt.level):
            block = ResBlocks(opt.depth, opt.dims[i] if i==0 else opt.dims[i-1], opt.dims[i], opt.ksizes[i])
            pool = nn.AvgPool2d(kernel_size=2)
            setattr(self, f'enc_block_{i}', block)
            setattr(self, f'enc_pool_{i}', pool)
    
    def forward(self, x):
        x = self.expand_dims(x)
        encoded = []
        for i in range(self.opt.level):
            block = getattr(self, f'enc_block_{i}')
            pool = getattr(self, f'enc_pool_{i}')
            x = block(x)
            encoded.append(x)
            x = pool(x)
        return x, encoded

class UNetEncoderFrequency(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.expand_dims = nn.Conv2d(2*opt.color_channels, opt.dims[0], kernel_size=1)
        for i in range(opt.level):
            block = ResBlocks(opt.depth, opt.dims[i] if i==0 else opt.dims[i-1], opt.dims[i], opt.ksizes[i])
            pool = nn.AvgPool2d(kernel_size=2)
            setattr(self, f'enc_block_{i}', block)
            setattr(self, f'enc_pool_{i}', pool)
    
    def forward(self, x):
        x = self.expand_dims(x)
        #encoded = []
        for i in range(self.opt.level):
            block = getattr(self, f'enc_block_{i}')
            pool = getattr(self, f'enc_pool_{i}')
            x = block(x)
            #encoded.append(x)
            x = pool(x)
        return x

class DomainMerger(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.depth_bottom = opt.depth_bottom
        dim = opt.dims[-1]
        self.cpe_spatial = ConditionalPositionalEncoding(dim)
        self.cpe_frequency = ConditionalPositionalEncoding(dim)
        for i in range(self.depth_bottom):
            enc = AxialTransformerBlock(dim, heads=opt.heads, mlp_ratio=opt.mlp_ratio)
            dec = AxialTransformerDecoderBlock(dim, heads=opt.heads, mlp_ratio=opt.mlp_ratio)
            setattr(self, f'enc_{i}', enc)
            setattr(self, f'dec_{i}', dec)
    
    def forward(self, x, x_f):
        x = self.cpe_spatial(x)
        x_f = self.cpe_frequency(x)
        for i in range(self.depth_bottom):
            enc = getattr(self, f'enc_{i}')
            x = enc(x)
        for i in range(self.depth_bottom):
            dec = getattr(self, f'dec_{i}')
            x = dec(x_f, x)
        return x

def UNetDecoder(nn.Module):
    pass

class DualDomainNet(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.encoder_spatial = UNetEncoderSpatial(opt)
        self.encoder_frequency = UNetEncoderFrequency(opt)
        self.domain_merger = DomainMerger(opt)

    def forward(self, x):



class UNet(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.expand_dims = nn.Conv2d(opt.color_channels, opt.dims[0], kernel_size=1)
        for i in range(opt.level):
            block = ResBlocks(opt.depth, opt.dims[i] if i==0 else opt.dims[i-1], opt.dims[i], opt.ksizes[i])
            pool = nn.AvgPool2d(kernel_size=2)
            setattr(self, f'enc_block_{i}', block)
            setattr(self, f'enc_pool_{i}', pool)
        
        block = ResBlocks(opt.depth, opt.dims[-1], opt.dims[-1], opt.ksizes[-1])
        setattr(self, f'bottom_block_{i}', block)

        for i in range(opt.level):
            idx1 = opt.level-1-i
            idx2 = opt.level-2-i if i!=opt.level-1 else 0
            block = ResBlocks(opt.depth, 2*opt.dims[idx1], opt.dims[idx2], opt.ksizes[idx2])
            setattr(self, f'dec_block_{i}', block)
        
        self.to_out = nn.Conv2d(opt.dims[0], opt.color_channels, opt.ksizes[0], stride=1, padding=opt.ksizes[0]//2)
    
    def forward(self, x):
        x = self.expand_dims(x)

        encoded = []
        for i in range(self.opt.level):
            block = getattr(self, f'enc_block_{i}')
            pool = getattr(self, f'enc_pool_{i}')
            x = block(x)
            encoded.append(x)
            x = pool(x)
        
        block = getattr(self, f'bottom_block_{i}')
        x = block(x)
    
        for i in range(self.opt.level):
            idx = self.opt.level-1-i
            block = getattr(self, f'dec_block_{i}')
            x = F.interpolate(x, size=(encoded[idx].size(2), encoded[idx].size(3)), mode='bilinear', align_corners=False)
            x = torch.cat([x, encoded[idx]], dim=1)
            x = block(x)
        x = self.to_out(x)

        return x