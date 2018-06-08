import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data
import model.data_loader as data_loader

from torchvision.models.inception import inception_v3

import numpy as np
from scipy.stats import entropy

import argparse
import logging
import os

import numpy as np
import torch
from torch.autograd import Variable
import utils
import model.data_loader as data_loader

from skimage import io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
from skimage import util
import torchvision.transforms.functional as F

import model.RESNET as Resnet
import model.srcnn as Srcnn
import model.fsrcnn as Fsrcnn
import model.densenet as Densenet
import model.densenet_shallow as Densenet_shallow
import model.des_size as Densenet_size
import model.drrn as Drrn
# base drrn_b1u9
import model.drrn_b1u9 as Drrn_b1u9
# filter size
import model.drrn_b1u9_filter_5 as Drrn_b1u9_filter_5
import model.drrn_b1u9_filter_7 as Drrn_b1u9_filter_7
# # of filters
import model.drrn_b1u9_filter_64 as Drrn_b1u9_filter_64
import model.drrn_b1u9_filter_32 as Drrn_b1u9_filter_32
# # of res units
import model.drrn_b1u15 as Drrn_b1u15
import model.drrn_b1u5 as Drrn_b1u5

import model.drrn_u9 as Drrn_u9
import model.drrn_dense as Drrn_dense

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='../data/4_3_srcnn', help="Directory containing the dataset")

parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")

def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs

    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    print("imgs: ", len(imgs))
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval();
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        f = torch.nn.Softmax()
        return f(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

if __name__ == '__main__':
    class IgnoreLabelDataset(torch.utils.data.Dataset):
        def __init__(self, orig):
            self.orig = orig

        def __getitem__(self, index):
            return self.orig[index][0]

        def __len__(self):
            return len(self.orig)

    import torchvision.datasets as dset
    import torchvision.transforms as transforms

#     cifar = dset.CIFAR10(root='data/', download=True,
#                              transform=transforms.Compose([
#                                  transforms.Scale(32),
#                                  transforms.ToTensor(),
#                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#                              ])
#     )
    
    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()     # use GPU is available

    cuda_id = 0
    
    dataloaders = data_loader.fetch_dataloader(['test'], args.data_dir, params)
    cifar = dataloaders['test']
    print(len(cifar))
    
#     IgnoreLabelDataset(cifar)

    print ("Calculating Inception Score...")
    print (inception_score(cifar, cuda=True, batch_size=32, resize=True, splits=10))
