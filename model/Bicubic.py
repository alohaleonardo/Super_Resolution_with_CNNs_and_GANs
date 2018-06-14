import torch
import torch.nn as nn
import math
import numpy as np
# from model.base_networks import ResnetBlock

# We use the Resnet_Block in base_networks.py

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
#         self.upscale = nn.functional.upsample(input, size=None, scale_factor=None, mode='nearest')


    def forward(self, x):
        out = nn.functional.upsample(x, size=None, scale_factor=2, mode='nearest')
#         out = x
#         residual = out
#         out = self.residual(out)
#         out = self.bn_mid(self.conv_mid(out))
#         out = torch.add(out,residual)
# #         out = self.upscale4x(out)
#         out = self.conv_output(out)
        return out


def loss_fn(outputs, labels):
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
    print (outputs.shape)
    print (labels.shape)
    result_by = torch.sum((outputs - labels) ** 2) / outputs.size()[0]
    return result_by


def accuracy(outputs, labels) :
    nume = np.max(outputs, axis = (1, 2, 3), keepdims = True)  #(N,)
    deno = np.sum((outputs.reshape(-1,3,64,64) - labels.reshape(-1,3,64,64))**2, axis = (1, 2, 3), keepdims = True)  # (N,)
    
    psnr = 10 * np.sum(np.log((nume*256)**2 / deno) / math.log(10)) / outputs.shape[0]
    return psnr


# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'accuracy': accuracy,
}

def calculate_psnr(outputs, labels) :
    psnr = np.log((np.max(outputs) / np.sum((outputs.reshape(-1, 3, 64, 64) - labels.reshape(-1, 3, 64, 64))**2) / outputs.shape[0])) / math.log(10)
    return psnr