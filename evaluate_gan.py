"""Evaluates the model"""

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
import model.drrn_modified as Drrn_modified
import model.gan as GAN
import model.gan_adv as GAN_adv
import model.gan_ssim as GAN_ssim
import model.gan_notv as GAN_notv
import model.cgan as CGAN
import model.cgan_finetune as CGAN_finetune
model_directory = {'des_size': Densenet_size, 'densenet_shallow': Densenet_shallow, 'densenet': Densenet, 'resnet': Resnet, 'srcnn': Srcnn, 'fsrcnn': Fsrcnn, 'drrn': Drrn, 'drrn_modified': Drrn_modified, 'gan': GAN, 'gan_adv': GAN_adv, 'gan_ssim': GAN_ssim, 'gan_notv': GAN_notv, 'cgan': CGAN, 'cgan_finetune': CGAN_finetune}


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='../data/4_3_test', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
parser.add_argument('--model', default=None)
parser.add_argument('--cuda', default=None)
parser.add_argument('--optim', default='adam')
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'

opt = parser.parse_args()
net = model_directory[opt.model]
from loss import GeneratorLoss, GeneratorLoss_adv, GeneratorLoss_ssim, GeneratorLoss_notv


def evaluate(netG, netD, loss_fn, dataloader, metrics, params, cuda_id):

    # set model to evaluation mode
    netG.eval()
    netD.eval()
    
    # summary for current eval loop
    summ = []

    count = 0
    # compute metrics over the dataset
    for data_batch, labels_batch in dataloader:

        # move to GPU if available
        if params.cuda:
            data_batch, labels_batch = data_batch.cuda(cuda_id, async=True), labels_batch.cuda(cuda_id, async=True)
        # fetch the next evaluation batch
        data_batch, labels_batch = Variable(data_batch), Variable(labels_batch)
        
        # compute model output
        output_batch = netG(data_batch)
        
#         output_batch_tensor = output_batch.data.cpu()
#         for i in output_batch_tensor:
#             #i /= np.max(i) * 2
#             #io.imsave(os.path.join('data/results/', '%d-evaluate-result.jpg' %count), i.reshape(128,128,3))
#             i = torch.clamp(i,0.0,1.0)
#             image = F.to_pil_image(i)
#             image.save("experiments/gan_ssim_model/test_results/" + "%d-evaluate-result.jpg" %count)
#             count += 1
         
        
        N, C, H, W = output_batch.shape
        mse_loss = torch.sum((output_batch * 255 - labels_batch * 255) ** 2) / N / C / H / W  # each photo, each channel
    
        # extract data from torch Variable, move to cpu, convert to numpy arrays
        output_batch = output_batch.data.cpu().numpy()
       
        labels_batch = labels_batch.data.cpu().numpy()

        # compute all metrics on this batch
        summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                         for metric in metrics}
        summary_batch['mse_loss'] = mse_loss.data[0]
        summ.append(summary_batch)
        
    # compute mean of all metrics in summary
    #print("")
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]} 
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)
    return metrics_mean


if __name__ == '__main__':
    """
        Evaluate the model on the test set.
    """
    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()     # use GPU is available

    cuda_id = 0
    #set 8 gpu
    if args.cuda != None:
        if args.cuda == 'cuda0':
            cuda_id = 0
        elif args.cuda == 'cuda1':
            cuda_id = 1
        elif args.cuda == 'cuda2':
            cuda_id = 2
        elif args.cuda == 'cuda3':
            cuda_id = 3
        elif args.cuda == 'cuda4':
            cuda_id = 4
        elif args.cuda == 'cuda5':
            cuda_id = 5
        elif args.cuda == 'cuda6':
            cuda_id = 6
        elif args.cuda == 'cuda7':
            cuda_id = 7
            
    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda: torch.cuda.manual_seed(230)
        
    # Get the logger
    utils.set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Creating the dataset...")

    # fetch dataloaders
    dataloaders = data_loader.fetch_dataloader(['test'], args.data_dir, params)
    test_dl = dataloaders['test']

    logging.info("- done.")

    # Define the model
#     model = net.Net(params).cuda() if params.cuda else net.Net(params)
    netG = net.Generator(4).cuda(cuda_id)
    netD = net.Discriminator().cuda(cuda_id)
    
    loss_fn = None
    if net == GAN or net == CGAN:
        loss_fn = GeneratorLoss().cuda(cuda_id)
    elif net == GAN_adv:
        loss_fn = GeneratorLoss_adv().cuda(cuda_id)
    elif net == GAN_ssim:
        loss_fn = GeneratorLoss_ssim().cuda(cuda_id)
    elif net == GAN_notv:
        loss_fn = GeneratorLoss_notv().cuda(cuda_id)
    
    metrics = net.metrics
    
    logging.info("Starting evaluation")

    # Reload weights from the saved file
    restore_path_g = os.path.join(args.model_dir, 'best_g' + '.pth.tar')
    restore_path_d = os.path.join(args.model_dir, 'best_d' + '.pth.tar')
    utils.load_checkpoint(restore_path_g, netG)
    utils.load_checkpoint(restore_path_d, netD)
#     utils.load_checkpoint(os.path.join(args.model_dir, args.restore_file + '.pth.tar'), model)

    # Evaluate
    test_metrics = evaluate(netG, netD, loss_fn, test_dl, metrics, params, cuda_id)
    save_path = os.path.join(args.model_dir, "metrics_test_{}.json".format(args.restore_file))
    utils.save_dict_to_json(test_metrics, save_path)

