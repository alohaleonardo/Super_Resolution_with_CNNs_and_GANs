import torch
import torch.nn as nn
import math
import numpy as np
# from model.base_networks import ResnetBlock

# We use the Resnet_Block in base_networks.py
class _Residual_Block(nn.Module):
    def __init__(self):
        super(_Residual_Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.in1 = nn.InstanceNorm2d(64, affine=True)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.in2 = nn.InstanceNorm2d(64, affine=True)

    def forward(self, x):
        identity_data = x
        output = self.relu(self.in1(self.conv1(x)))
        output = self.in2(self.conv2(output))
        output = torch.add(output,identity_data)
        return output 

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv_input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, stride=1, padding=2, bias=False)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        
        self.residual = self.make_layer(_Residual_Block, 16)

        self.conv_mid = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_mid = nn.InstanceNorm2d(64, affine=True)

#         self.upscale4x = nn.Sequential(
#             nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.PixelShuffle(2),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.PixelShuffle(2),
#             nn.LeakyReLU(0.2, inplace=True),
#         )

        self.conv_output = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=9, stride=1, padding=0, bias=False)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.conv_input(x))
        residual = out
        out = self.residual(out)
        out = self.bn_mid(self.conv_mid(out))
        out = torch.add(out,residual)
#         out = self.upscale4x(out)
        out = self.conv_output(out)
        return out

# class _NetD(nn.Module):
#     def __init__(self):
#         super(_NetD, self).__init__()

#         self.features = nn.Sequential(
        
#             # input is (3) x 96 x 96
#             nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),

#             # state size. (64) x 96 x 96
#             nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),            
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.2, inplace=True),

#             # state size. (64) x 96 x 96
#             nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),            
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2, inplace=True),
            
#             # state size. (64) x 48 x 48
#             nn.Conv2d(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2, inplace=True),

#             # state size. (128) x 48 x 48
#             nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2, inplace=True),

#             # state size. (256) x 24 x 24
#             nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2, inplace=True),

#             # state size. (256) x 12 x 12
#             nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),            
#             nn.BatchNorm2d(512),
#             nn.LeakyReLU(0.2, inplace=True),

#             # state size. (512) x 12 x 12
#             nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False),            
#             nn.BatchNorm2d(512),
#             nn.LeakyReLU(0.2, inplace=True),
#         )

#         self.LeakyReLU = nn.LeakyReLU(0.2, inplace=True)
#         self.fc1 = nn.Linear(512 * 6 * 6, 1024)
#         self.fc2 = nn.Linear(1024, 1)
#         self.sigmoid = nn.Sigmoid()

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 m.weight.data.normal_(0.0, 0.02)
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.normal_(1.0, 0.02)
#                 m.bias.data.fill_(0)

#     def forward(self, input):

#         out = self.features(input)

#         # state size. (512) x 6 x 6
#         out = out.view(out.size(0), -1)

#         # state size. (512 x 6 x 6)
#         out = self.fc1(out)

#         # state size. (1024)
#         out = self.LeakyReLU(out)

#         out = self.fc2(out)
#         out = self.sigmoid(out)
#         return out.view(-1, 1).squeeze(1)


# This part is copied from SRCNN

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
#     print (outputs.shape)
#     print (labels.shape)
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