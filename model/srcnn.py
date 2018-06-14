import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.base_networks import *
import skimage as sk
import math

import pytorch_ssim as ps
from torch.autograd import Variable
from skimage.measure import compare_psnr, compare_ssim

class Net(nn.Module):

    def __init__(self, params):
        super(Net, self).__init__()
        self.num_channels = params.num_channels
        self.dropout_rate = params.dropout_rate
        
        self.layers = torch.nn.Sequential(
            ConvBlock(3, self.num_channels, 9, 1, 4, norm=None), # 144*144*64 # conv->batchnorm->activation
            ConvBlock(self.num_channels, self.num_channels // 2, 1, 1, 0, norm=None), # 144*144*32
            ConvBlock(self.num_channels // 2, 3, 5, 1, 2, activation=None, norm=None) # 144*144*1
        )

    def forward(self, s):
        out = self.layers(s)
        return out

def loss_fn(outputs, labels):
    N, C, H, W = outputs.shape
        
    mse_loss = torch.sum((outputs - labels) ** 2) / N / C   # each photo, each channel
    mse_loss *= 255 * 255
    mse_loss /= H * W  
    # average loss on each pixel(0-255)
    return mse_loss

def accuracy(outputs, labels):
    N, _, _, _ = outputs.shape
    psnr = 0
    for i in range(N):
        psnr += compare_psnr(labels[i],outputs[i])
    return psnr / N

#     N, C, H, W = outputs.shape
    
#     nume = np.max(outputs, axis = (1, 2, 3), keepdims = True)  #(N,)
#     deno = np.sum((outputs.reshape(-1,3,144,144) - labels.reshape(-1,3,144,144))**2, axis = (1, 2, 3), keepdims = True) / C
#     deno *=  255 * 255 / H / W   # (N,)  range from 0-255, pixel avg
    
#     psnr = (nume * 255) ** 2 / deno # (N,)
#     psnr = np.log(psnr)
#     psnr = 10 * np.sum(psnr)
#     psnr /= math.log(10) * N

#     return psnr

def ssim(outputs, labels) :
    N, _, _, _ = outputs.shape
    ssim = 0
    for i in range(N):
        
        ssim += compare_ssim(labels[i],outputs[i], win_size=3, multichannel=True)
    return ssim / N
    
#     if torch.cuda.is_available():
#         outputs = Variable( torch.from_numpy(outputs)).cuda()
#         labels = Variable( torch.from_numpy(labels)).cuda()

#     ssim = ps.ssim(outputs, labels)
#     return ssim
    

# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'PSNR': accuracy,
    'SSIM': ssim,
    # could add more metrics such as accuracy for each token type
}