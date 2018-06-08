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

# class DRRN(nn.Module
class Net(nn.Module):
    def __init__(self, params):
        super(Net, self).__init__()
        self.input = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.output = nn.Conv2d(in_channels=128, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

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
        prev_out = []
        for i in range(5):
            if i == 0:
                out = self.conv2(self.relu(self.conv1(self.relu(out))))
                out = torch.add(out, inputs)
                prev_out.append(out)
                continue
            out = self.conv2(self.relu(self.conv1(self.relu(out))))
            out = torch.add(torch.add(out, inputs), prev_out[-1])
            prev_out.append(out)
            
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






# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from math import sqrt
# from torch.autograd import Variable
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from model.base_networks import *
# import skimage as sk
# import math

# import pytorch_ssim as ps
# from torch.autograd import Variable
# from skimage.measure import compare_psnr, compare_ssim

# # class DRRN(nn.Module
# class Net(nn.Module):
#     def __init__(self, params):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
#         self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
#         self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
#         self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
#         self.conv5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
#         self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
#         self.conv7 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
#         self.conv8 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
#         self.conv9 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
#         self.conv10 = nn.Conv2d(in_channels=128, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
#         self.relu = nn.ReLU(inplace=True)

#         nn.init.kaiming_normal_(self.conv1.weight)
#         nn.init.kaiming_normal_(self.conv2.weight)
#         nn.init.kaiming_normal_(self.conv3.weight)
#         nn.init.kaiming_normal_(self.conv4.weight)
#         nn.init.kaiming_normal_(self.conv5.weight)
#         nn.init.kaiming_normal_(self.conv6.weight)
#         nn.init.kaiming_normal_(self.conv7.weight)
#         nn.init.kaiming_normal_(self.conv8.weight)
#         nn.init.kaiming_normal_(self.conv9.weight)
#         nn.init.kaiming_normal_(self.conv10.weight)
        
# #         # weights initialization
# #         for m in self.modules():
# #             if isinstance(m, nn.Conv2d):
# #                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
# #                 m.weight.data.normal_(0, sqrt(2. / n))

#     def forward(self, x):
#         res = x
#         out1 = self.conv1(self.relu(res))
#         out2 = self.conv2(self.relu(out1))
#         out3 = self.conv3(self.relu(out1 + out2))
#         out4 = self.conv4(self.relu(out1 + out2 + out3))
#         out5 = self.conv5(self.relu(out1 + out2 + out3 + out4))
#         out6 = self.conv6(self.relu(out1 + out2 + out3 + out4 + out5))
#         out7 = self.conv7(self.relu(out1 + out2 + out3 + out4 + out5 + out6))
#         out8 = self.conv8(self.relu(out1 + out2 + out3 + out4 + out5 + out6 + out7))
#         out9 = self.conv9(self.relu(out1 + out2 + out3 + out4 + out5 + out6 + out7 + out8))
#         out10 = self.conv10(self.relu(out9))
        
# #         out = out10
# #         for _ in range(5):
# #             out = self.conv2(self.relu(self.conv1(self.relu(out))))
# #             out = torch.add(out, inputs)

# #         out = self.output(self.relu(out))
# #         out = torch.add(out, residual)
#         return out10

# def loss_fn(outputs, labels):
#     N, C, H, W = outputs.shape
        
#     mse_loss = torch.sum((outputs - labels) ** 2) / N / C   # each photo, each channel
#     mse_loss *= 255 * 255
#     mse_loss /= H * W  
#     # average loss on each pixel(0-255)
#     return mse_loss

# def accuracy(outputs, labels):
#     N, _, _, _ = outputs.shape
#     psnr = 0
#     for i in range(N):
#         psnr += compare_psnr(labels[i],outputs[i])
#     return psnr / N

# def ssim(outputs, labels) :
#     N, _, _, _ = outputs.shape
#     ssim = 0
#     for i in range(N):
        
#         ssim += compare_ssim(labels[i],outputs[i], win_size=3, multichannel=True)
#     return ssim / N   

# # maintain all metrics required in this dictionary- these are used in the training and evaluation loops
# metrics = {
#     'PSNR': accuracy,
#     'SSIM': ssim,
#     # could add more metrics such as accuracy for each token type
# }