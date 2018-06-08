import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from torch.autograd import Variable
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

class Dense_Block(nn.Module):
    def __init__(self, channel_in):
        super(Dense_Block, self).__init__()

        self.relu = nn.PReLU()
        self.conv1 = nn.Conv2d(in_channels=channel_in, out_channels=16, kernel_size=9, stride=1, padding=4)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=9, stride=1, padding=4)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=9, stride=1, padding=4)
        self.conv4 = nn.Conv2d(in_channels=48, out_channels=16, kernel_size=9, stride=1, padding=4)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.conv6 = nn.Conv2d(in_channels=80, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.conv7 = nn.Conv2d(in_channels=96, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.conv8 = nn.Conv2d(in_channels=112, out_channels=16, kernel_size=5, stride=1, padding=2)
        
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.conv3.weight)
        nn.init.kaiming_normal_(self.conv4.weight)
        nn.init.kaiming_normal_(self.conv5.weight)
        nn.init.kaiming_normal_(self.conv6.weight)
        nn.init.kaiming_normal_(self.conv7.weight)
        nn.init.kaiming_normal_(self.conv8.weight)
        
        self.ins1 = nn.InstanceNorm2d(16)
        
    def forward(self, x):
        
        conv1 = self.relu(self.ins1(self.conv1(x))) # 16

        conv2 = self.relu(self.ins1(self.conv2(conv1))) # 16
        cout2_dense = self.relu(torch.cat([conv1,conv2], 1))

        conv3 = self.relu(self.ins1(self.conv3(cout2_dense)))
        cout3_dense = self.relu(torch.cat([conv1,conv2,conv3], 1))

        conv4 = self.relu(self.ins1(self.conv4(cout3_dense)))
        cout4_dense = self.relu(torch.cat([conv1,conv2,conv3,conv4], 1))

        conv5 = self.relu(self.ins1(self.conv5(cout4_dense)))
        cout5_dense = self.relu(torch.cat([conv1,conv2,conv3,conv4,conv5], 1))

        conv6 = self.relu(self.ins1(self.conv6(cout5_dense)))
        cout6_dense = self.relu(torch.cat([conv1,conv2,conv3,conv4,conv5,conv6], 1))

        conv7 = self.relu(self.ins1(self.conv7(cout6_dense)))
        cout7_dense = self.relu(torch.cat([conv1,conv2,conv3,conv4,conv5,conv6,conv7], 1))

        conv8 = self.relu(self.ins1(self.conv8(cout7_dense)))
        cout8_dense = self.relu(torch.cat([conv1,conv2,conv3,conv4,conv5,conv6,conv7,conv8], 1))
        
        
# class DRRN(nn.Module
class Net(nn.Module):
    def __init__(self, params):
        super(Net, self).__init__()
        self.input = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.output = nn.Conv2d(in_channels=128, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.dense = Dense_

        nn.init.kaiming_normal_(self.input.weight)
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.output.weight)

#         # weights initialization
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, sqrt(2. / n))

    def forward(self, x):
        residual = x
        inputs = self.input(self.relu(x))
        out = inputs
        for _ in range(5):
            out = self.conv2(self.relu(self.conv1(self.relu(out))))
            out = torch.add(out, inputs)

        out = self.output(self.relu(out))
        out = torch.add(out, residual)
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

def ssim(outputs, labels) :
    N, _, _, _ = outputs.shape
    ssim = 0
    for i in range(N):
        
        ssim += compare_ssim(labels[i],outputs[i], win_size=3, multichannel=True)
    return ssim / N   

# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'PSNR': accuracy,
    'SSIM': ssim,
    # could add more metrics such as accuracy for each token type
}