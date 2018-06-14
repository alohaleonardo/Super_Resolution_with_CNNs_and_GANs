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
        self.conv1 = nn.Conv2d(in_channels=channel_in, out_channels=16, kernel_size=11, stride=1, padding=5)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=11, stride=1, padding=5)
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
        self.ins2 = nn.InstanceNorm2d(32)
        self.ins3 = nn.InstanceNorm2d(48)
        self.ins4 = nn.InstanceNorm2d(64)
        self.ins5 = nn.InstanceNorm2d(80)
        self.ins6 = nn.InstanceNorm2d(96)
        self.ins7 = nn.InstanceNorm2d(112)
        self.ins8 = nn.InstanceNorm2d(16)
        
    def forward(self, x):
        
        conv1 = self.relu(self.ins1(self.conv1(x))) # 16

        conv2 = self.relu(self.ins2(self.conv2(conv1))) # 16
        cout2_dense = self.relu(torch.cat([conv1,conv2], 1))

        conv3 = self.relu(self.ins3(self.conv3(cout2_dense)))
        cout3_dense = self.relu(torch.cat([conv1,conv2,conv3], 1))

        conv4 = self.relu(self.ins4(self.conv4(cout3_dense)))
        cout4_dense = self.relu(torch.cat([conv1,conv2,conv3,conv4], 1))

        conv5 = self.relu(self.ins5(self.conv5(cout4_dense)))
        cout5_dense = self.relu(torch.cat([conv1,conv2,conv3,conv4,conv5], 1))

        conv6 = self.relu(self.ins6(self.conv6(cout5_dense)))
        cout6_dense = self.relu(torch.cat([conv1,conv2,conv3,conv4,conv5,conv6], 1))

        conv7 = self.relu(self.ins7(self.conv7(cout6_dense)))
        cout7_dense = self.relu(torch.cat([conv1,conv2,conv3,conv4,conv5,conv6,conv7], 1))

        conv8 = self.relu(self.ins8(self.conv8(cout7_dense)))
        cout8_dense = self.relu(torch.cat([conv1,conv2,conv3,conv4,conv5,conv6,conv7,conv8], 1))
        
#         conv1 = self.relu(self.conv1(x)) # 16

#         conv2 = self.relu(self.conv2(conv1)) # 16
#         cout2_dense = self.relu(torch.cat([conv1,conv2], 1))

#         conv3 = self.relu(self.conv3(cout2_dense))
#         cout3_dense = self.relu(torch.cat([conv1,conv2,conv3], 1))

#         conv4 = self.relu(self.conv4(cout3_dense))
#         cout4_dense = self.relu(torch.cat([conv1,conv2,conv3,conv4], 1))

#         conv5 = self.relu(self.conv5(cout4_dense))
#         cout5_dense = self.relu(torch.cat([conv1,conv2,conv3,conv4,conv5], 1))

#         conv6 = self.relu(self.conv6(cout5_dense))
#         cout6_dense = self.relu(torch.cat([conv1,conv2,conv3,conv4,conv5,conv6], 1))

#         conv7 = self.relu(self.conv7(cout6_dense))
#         cout7_dense = self.relu(torch.cat([conv1,conv2,conv3,conv4,conv5,conv6,conv7], 1))

#         conv8 = self.relu(self.conv8(cout7_dense))
#         cout8_dense = self.relu(torch.cat([conv1,conv2,conv3,conv4,conv5,conv6,conv7,conv8], 1))

        return cout8_dense

class Net(nn.Module):
    def __init__(self,params):
        super(Net, self).__init__()
        
        self.relu = nn.ReLU()
        self.lowlevel = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bottleneck = nn.Conv2d(in_channels=640, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False)
        self.reconstruction = nn.Conv2d(in_channels=256, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        self.denseblock1 = self.make_layer(_Dense_Block, 128)
        self.denseblock2 = self.make_layer(_Dense_Block, 256)
        self.denseblock3 = self.make_layer(_Dense_Block, 384)
        self.denseblock4 = self.make_layer(_Dense_Block, 512)
        self.denseblock5 = self.make_layer(_Dense_Block, 640)
        self.denseblock6 = self.make_layer(_Dense_Block, 768)
        self.denseblock7 = self.make_layer(_Dense_Block, 896)
        self.denseblock8 = self.make_layer(_Dense_Block, 1024)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=2, stride=2, padding=0, bias=False),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=2, stride=2, padding=0, bias=False),
            nn.ReLU()
        )
        
        nn.init.kaiming_normal_(self.lowlevel.weight)
        nn.init.kaiming_normal_(self.bottleneck.weight)
        nn.init.kaiming_normal_(self.reconstruction.weight)
        
        
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#             if isinstance(m, nn.ConvTranspose2d):
#                 c1, c2, h, w = m.weight.data.size()
#                 weight = get_upsample_filter(h)
#                 m.weight.data = weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)
#                 if m.bias is not None:
#                     m.bias.data.zero_()
       
    def make_layer(self, block, channel_in):
        layers = []
        layers.append(block(channel_in))
        return nn.Sequential(*layers)

    def forward(self, x):    
        residual = self.relu(self.lowlevel(x))

        out = self.denseblock1(residual)
        concat = torch.cat([residual,out], 1)

        out = self.denseblock2(concat)
        concat = torch.cat([concat,out], 1)

        out = self.denseblock3(concat)
        concat = torch.cat([concat,out], 1)
        
        out = self.denseblock4(concat)
        concat = torch.cat([concat,out], 1)
        
#         out = self.denseblock5(concat)
#         concat = torch.cat([concat,out], 1)
        
#         out = self.denseblock6(concat)
#         concat = torch.cat([concat,out], 1)
        
#         out = self.denseblock7(concat)
#         concat = torch.cat([concat,out], 1)
        
#         out = self.denseblock8(concat)
#         out = torch.cat([concat,out], 1)

        out = self.bottleneck(concat)

        out = self.deconv(out)

        out = self.reconstruction(out)  # (64, 3, 144, 144)
        
        return out



def loss_fn(outputs, labels):
    #print('this is outputs', outputs.shape) # 2,3,128,128
    #print('this is labels', labels.shape) # 2,3,128,128
    """
    Compute the cross entropy loss given outputs and labels.
    Args:
        outputs: (Variable) dimension batch_size x 6 - output of the model
        labels: (Variable) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]
    Returns:
        loss (Variable): cross entropy loss for all images in the batch
    Note: you may use a standard loss function from http://pytorch.org/docs/master/nn.html#loss-functions. This example
          demonstrates how you can easily define a custom loss function.
    """
    N, C, H, W = outputs.shape
    
#     outputs = unnormalize(outputs, mean=[0.51371954, 0.40949144, 0.35572536], std= [0.2926419,  0.26180502, 0.25512055])
    # check if we normalize label images #labels = unnormalize(labels, mean=[0.53459634,0.39673596,0.33788489], std= [0.29101071,0.26140346,0.25485687])
        
    mse_loss = torch.sum((outputs - labels) ** 2) / N / C   # each photo, each channel
    mse_loss *= 255 * 255
    mse_loss /= H * W  
    # average loss on each pixel(0-255)
    return mse_loss


def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all images.
    Args:
        outputs: (np.ndarray) dimension batch_size x 6 - log softmax output of the model
        labels: (np.ndarray) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]
    Returns: (float) accuracy in [0,1]
    """
    N, C, H, W = outputs.shape
#     outputs = unnormalize(outputs, mean=[0.51371954, 0.40949144, 0.35572536], std= [0.2926419,  0.26180502, 0.25512055])
    # check if we normalize label images #labels = unnormalize(labels, mean=[0.53459634,0.39673596,0.33788489], std= [0.29101071,0.26140346,0.25485687])
    
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



def unnormalize(image, mean, std):
    '''
    image(N, 3, H, W)
    mean(3,)
    std(3,)
    '''
    for i in range(3):
        image[:,i,:,:] = image[:,i,:,:] * std[i] + mean[i]
    return image