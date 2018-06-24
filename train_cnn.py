"""Train the model"""

import argparse
import logging
import os

import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm

import utils
from torch.nn.utils import clip_grad_norm
from torch.optim import lr_scheduler
import model.data_loader as data_loader
from evaluate_cnn import evaluate
import torchvision.transforms.functional as F
from skimage import io
from logger import Logger
import matplotlib.pyplot as plt
plt.switch_backend('agg')

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
parser.add_argument('--data_dir', default='../data/cnn_faces', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
parser.add_argument('--model', default=None)
parser.add_argument('--cuda', default=None)
parser.add_argument('--optim', default='adam')
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'


opt = parser.parse_args()
net = model_directory[opt.model]

global_loss = []


def train(model, optimizer, loss_fn, dataloader, metrics, params, logger, epoch, cuda_id):

    # set model to training mode
    model.train()
    #print('this the type of the model', type(model))

    # summary for current training loop and a running average object for loss
    summ = []
    loss_avg = utils.RunningAverage()
    
    current_learning_rate = optimizer.defaults['lr']

    # Use tqdm for progress bar
    with tqdm(total=len(dataloader)) as t:
        for i, (train_batch, labels_batch) in enumerate(dataloader):
            # move to GPU if available
            if params.cuda:
                train_batch, labels_batch = train_batch.cuda(cuda_id, async=True), labels_batch.cuda(cuda_id, async=True)
            # convert to torch Variables
            train_batch, labels_batch = Variable(train_batch), Variable(labels_batch)

            # compute model output and loss
            output_batch = model(train_batch)
            loss = loss_fn(output_batch, labels_batch)

            # clear previous gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss.backward()
#             clip_grad_norm(model.parameters(), 0.01 / current_learning_rate)

            # performs updates using calculated gradients
            optimizer.step()
            
            # Evaluate summaries only once in a while
            if i % params.save_summary_steps == 0:
                # extract data from torch Variable, move to cpu, convert to numpy arrays
                output_batch = output_batch.data.cpu().numpy()
                labels_batch = labels_batch.data.cpu().numpy()

                # compute all metrics on this batch
                summary_batch = {metric:metrics[metric](output_batch, labels_batch)
                                 for metric in metrics}
                summary_batch['loss'] = loss.item()
                summ.append(summary_batch)

            global_loss.append(loss.item())
    
            loss_avg.update(loss.item())

            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()

    # compute mean of all metrics in summary
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)


def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, loss_fn, metrics, params, model_dir,
                       restore_file=None, cuda_id = 0):
    
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(args.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)

    best_val_acc = 0.0
    
    '''
    # train add logger,epoch two parameters
    '''
    logger = Logger('./logs')
    scheduler = lr_scheduler.StepLR(optimizer, step_size = 10, gamma=1)

    for epoch in range(params.num_epochs):
        # Run one epoch
#         scheduler.step()
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        train(model, optimizer, loss_fn, train_dataloader, metrics, params, logger, epoch, cuda_id)
        
        
        # Evaluate for one epoch on validation set
        val_metrics = evaluate(model, loss_fn, val_dataloader, metrics, params, cuda_id)

        val_acc = val_metrics['PSNR']
        is_best = val_acc>=best_val_acc

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict' : optimizer.state_dict()},
                               is_best=is_best,
                               checkpoint=model_dir)

        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best accuracy")
            best_val_acc = val_acc

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(model_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json(val_metrics, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(model_dir, "metrics_val_last_weights.json")
        utils.save_dict_to_json(val_metrics, last_json_path)
    
    plt.plot(global_loss)
    plt.savefig("final loss.jpg")
        
        

if __name__ == '__main__':

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()
    
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

    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    # fetch dataloaders
    dataloaders = data_loader.fetch_dataloader(['train', 'val'], args.data_dir, params)
    train_dl = dataloaders['train']
    val_dl = dataloaders['val']
    
    logging.info("- done.")
    
    # Define the model and optimizer

    model = net.Net(params).cuda(cuda_id) if params.cuda else net.Net(params)

    if args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
    elif args.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=params.learning_rate, momentum=0.9)

    # fetch loss function and metrics
    loss_fn = net.loss_fn
    metrics = net.metrics

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(model, train_dl, val_dl, optimizer, loss_fn, metrics, params, args.model_dir,
                       args.restore_file, cuda_id=cuda_id)

    print("finish training and evaluating!")
    