import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.base_networks import *
import skimage as sk
import math

import pytorch_ssim as ps
from torch.autograd import Variable

def get_upsample_filter(size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filter = (1 - abs(og[0] - center) / factor) * \
             (1 - abs(og[1] - center) / factor)
    return torch.from_numpy(filter).float()

class _Dense_Block(nn.Module):
    def __init__(self, channel_in):
        super(_Dense_Block, self).__init__()

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=channel_in, out_channels=128, kernel_size=11, stride=1, padding=5)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(in_channels=192, out_channels=32, kernel_size=7, stride=1, padding=3)
        
#         ConvBlock(3, self.num_channels, 9, 1, 4, norm=None), # 144*144*64 # conv->batchnorm->activation
#         ConvBlock(self.num_channels, self.num_channels // 2, 1, 1, 0, norm=None), # 144*144*32
#         ConvBlock(self.num_channels // 2, 3, 5, 1, 2, activation=None, norm=None) # 144*144*1
        
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.conv3.weight)
        
        self.ins1 = nn.InstanceNorm2d(16)
        
    def forward(self, x):
        
        conv1 = self.relu(self.conv1(x)) # 16

        conv2 = self.relu(self.conv2(conv1)) # 16
        cout2_dense = self.relu(torch.cat([conv1,conv2], 1))

        conv3 = self.relu(self.conv3(cout2_dense))
        cout3_dense = self.relu(torch.cat([conv1,conv2,conv3], 1))

        return cout3_dense

class Net(nn.Module):
    def __init__(self,params):
        super(Net, self).__init__()
        
        self.relu = nn.ReLU()
        #self.lowlevel = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=9, stride=1, padding=4)
        self.bottleneck = nn.Conv2d(in_channels=224, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
        self.reconstruction = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        self.denseblock1 = self.make_layer(_Dense_Block, 3)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=2, stride=2, padding=0, bias=False),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=2, stride=2, padding=0, bias=False),
            nn.ReLU()
        )
        
#         nn.init.kaiming_normal_(self.lowlevel.weight)
        nn.init.kaiming_normal_(self.bottleneck.weight)
        nn.init.kaiming_normal_(self.reconstruction.weight)
        
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=11, stride=1, padding=5)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, stride=1, padding=3)
        
        
    def make_layer(self, block, channel_in):
        layers = []
        layers.append(block(channel_in))
        return nn.Sequential(*layers)

    def forward(self, x):    
        #residual = self.relu(self.lowlevel(x))
        # des1
#         out = self.denseblock1(x)
# #         concat = torch.cat([residual,out], 1)
        
#         out = self.bottleneck(out)
        
        out = self.deconv(x)
        out = self.relu(self.conv1(out))
        out = self.relu(self.conv2(out))
        out = self.relu(self.conv3(out))
        
        #out = self.reconstruction(out)  # (64, 3, 144, 144)
        
        return out


def loss_fn(outputs, labels):
    N, C, H, W = outputs.shape
        
    mse_loss = torch.sum((outputs - labels) ** 2) / N / C   # each photo, each channel
    mse_loss *= 255 * 255
    mse_loss /= H * W  
    # average loss on each pixel(0-255)
    return mse_loss


def accuracy(outputs, labels):
    N, C, H, W = outputs.shape
    
    nume = np.max(outputs, axis = (1, 2, 3), keepdims = True)  #(N,)
    deno = np.sum((outputs.reshape(-1,3,144,144) - labels.reshape(-1,3,144,144))**2, axis = (1, 2, 3), keepdims = True) / C
    deno *=  255 * 255 / H / W   # (N,)  range from 0-255, pixel avg
    
    psnr = (nume * 255) ** 2 / deno # (N,)
    psnr = np.log(psnr)
    psnr = 10 * np.sum(psnr) 
    psnr /= math.log(10) * N
    #print(outputs.shape)
    #print(psnr)
    
    return psnr


def ssim(outputs, labels) :
    if torch.cuda.is_available():
        outputs = Variable( torch.from_numpy(outputs)).cuda()
        labels = Variable( torch.from_numpy(labels)).cuda()
    #print(outputs.size)
    ssim = ps.ssim(outputs, labels)
    #print('ssim')
    #print(ssim)
    return ssim
    

# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'PSNR': accuracy,
    'SSIM': ssim,
    # could add more metrics such as accuracy for each token type
}