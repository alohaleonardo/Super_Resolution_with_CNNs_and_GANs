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

model_directory = {'des_size': Densenet_size, 'densenet_shallow': Densenet_shallow, 'densenet': Densenet, 'resnet': Resnet, 'srcnn': Srcnn, 'fsrcnn': Fsrcnn, 'drrn': Drrn, 'drrn_b1u9': Drrn_b1u9, 'drrn_b1u9_filter_5': Drrn_b1u9_filter_5, 'drrn_b1u9_filter_7': Drrn_b1u9_filter_7, 'drrn_b1u9_filter_64': Drrn_b1u9_filter_64, 'drrn_b1u9_filter_32': Drrn_b1u9_filter_32, 'drrn_b1u15': Drrn_b1u15, 'drrn_b1u5': Drrn_b1u5, 'drrn_u9': Drrn_u9, 'drrn_dense': Drrn_dense }

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='../data/4_3', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
parser.add_argument('--model', default='densenet')
parser.add_argument('--cuda', default=None)
parser.add_argument('--optim', default='adam')
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
                     containing weights to load")

opt = parser.parse_args()
net = model_directory[opt.model]

def evaluate(model, loss_fn, dataloader, metrics, params, cuda_id):

    # set model to evaluation mode
    model.eval()

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
        output_batch = model(data_batch)
        # outputbatch -> image-> save
        
#         output_batch_tensor = output_batch.data.cpu()
#         for i in output_batch_tensor:
#             #i /= np.max(i) * 2
#             #io.imsave(os.path.join('data/results/', '%d-evaluate-result.jpg' %count), i.reshape(128,128,3))
#             i = torch.clamp(i,0.0,1.0)
#             image = F.to_pil_image(i)
#             image.save("experiments/drrn_b1u9_model/test_results/" + "%d-evaluate-result.jpg" %count)
#             count += 1
         
        loss = loss_fn(output_batch, labels_batch)

        # extract data from torch Variable, move to cpu, convert to numpy arrays
        output_batch = output_batch.data.cpu().numpy()
       
        labels_batch = labels_batch.data.cpu().numpy()

        # compute all metrics on this batch
        summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                         for metric in metrics}
        summary_batch['loss'] = loss.item()
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
    model = net.Net(params).cuda(cuda_id) if params.cuda else net.Net(params)
    
    loss_fn = net.loss_fn
    metrics = net.metrics
    
    logging.info("Starting evaluation")

    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(args.model_dir, args.restore_file + '.pth.tar'), model)

    # Evaluate
    test_metrics = evaluate(model, loss_fn, test_dl, metrics, params, cuda_id)
    save_path = os.path.join(args.model_dir, "metrics_test_{}.json".format(args.restore_file))
    utils.save_dict_to_json(test_metrics, save_path)

