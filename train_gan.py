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
import model.data_loader as data_loader
from evaluate import evaluate
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
# if net == GAN:
#     from net import Generator, Discriminator
from loss import GeneratorLoss, GeneratorLoss_adv, GeneratorLoss_ssim, GeneratorLoss_notv

global_loss_g = []
global_loss_d = []

def train(netG, netD, optimG, optimD, loss_fn, dataloader, metrics, params, cuda_id):
    
    # set model to training mode
    netG.train()
    netD.train()

    # summary for current training loop and a running average object for loss
    summ = []
    g_loss_avg = utils.RunningAverage()
    d_loss_avg = utils.RunningAverage()
    mse_loss_avg = utils.RunningAverage()
    
    # Use tqdm for progress bar
    with tqdm(total=len(dataloader)) as t:
        for i, (train_batch, labels_batch) in enumerate(dataloader):
            # move to GPU if available
            if params.cuda:
                train_batch, labels_batch = train_batch.cuda(cuda_id, async=True), labels_batch.cuda(cuda_id, async=True)
            # convert to torch Variables
            train_batch, labels_batch = Variable(train_batch), Variable(labels_batch)

            # update D network
            real_img = labels_batch
            fake_img = netG(train_batch)
            
            netD.zero_grad()
            real_out = netD(real_img).mean()
            fake_out = netD(fake_img).mean()
            d_loss = 1 - real_out + fake_out
            d_loss.backward(retain_graph=True)
            optimD.step()

            # update G network
            netG.zero_grad()
#             print("real_img", real_img)
            g_loss = loss_fn(fake_out, fake_img, real_img)
            g_loss.backward()
            optimG.step()
            fake_img = netG(train_batch)
            fake_out = netD(fake_img).mean()

            
            g_loss = loss_fn(fake_out, fake_img, real_img)
            d_loss = 1 - real_out + fake_out
            N, C, H, W = real_img.shape
            mse_loss = torch.sum((real_img * 255 - fake_img * 255) ** 2) / N / C / H / W  # each photo, each channel
            
            # Evaluate summaries only once in a while
            if i % params.save_summary_steps == 0:
                # extract data from torch Variable, move to cpu, convert to numpy arrays
                output_batch = real_img.data.cpu().numpy()
                labels_batch = fake_img.data.cpu().numpy()

                # compute all metrics on this batch
                summary_batch = {metric:metrics[metric](output_batch, labels_batch)
                                 for metric in metrics}               
                summary_batch['g_loss'] = g_loss.item()
                summary_batch['d_loss'] = d_loss.item()
                summary_batch['mse_loss'] = mse_loss.item()
                summ.append(summary_batch)
                

            global_loss_g.append(g_loss.item())
            global_loss_d.append(d_loss.item())
    
            g_loss_avg.update(g_loss.item())
            d_loss_avg.update(d_loss.item())
            mse_loss_avg.update(mse_loss.item())
            t.set_postfix(g_loss='{:05.3f}'.format(g_loss_avg()),d_loss='{:05.3f}'.format(d_loss_avg()),mse_loss='{:05.3f}'.format(mse_loss_avg()))
            t.update()
            
    # compute mean of all metrics in summary
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)


def train_and_evaluate(netG, netD, train_dataloader, val_dataloader, optimG, optimD, loss_fn, metrics, params, model_dir,
                       restore_file=None, cuda_id=0):
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path_g = os.path.join(args.model_dir, 'best_g' + '.pth.tar')
        restore_path_d = os.path.join(args.model_dir, 'best_d' + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path_g))
        utils.load_checkpoint(restore_path_g, netG, optimG)
        utils.load_checkpoint(restore_path_d, netD, optimD)

    best_val_acc = 0.0
      # train add logger,epoch two parameters 
#     logger = Logger('./logs')

    for epoch in range(params.num_epochs):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        train(netG, netD, optimG, optimD, loss_fn, train_dataloader, metrics, params, cuda_id)
                
        # Evaluate for one epoch on validation set
        val_metrics = evaluate(netG, netD, loss_fn, val_dataloader, metrics, params, cuda_id)
        #print ('after val --------')
        
        val_acc = val_metrics['PSNR']
        is_best = val_acc>=best_val_acc

        #Save weights
        # save G
        flag = 'G'
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': netG.state_dict(),
                               'optim_dict' : optimG.state_dict()},
                               is_best=is_best,
                               checkpoint=model_dir, flag=flag)
        flag = 'D'
        # save D
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': netD.state_dict(),
                               'optim_dict' : optimD.state_dict()},
                               is_best=is_best,
                               checkpoint=model_dir, flag=flag)

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
        
        if epoch % 100 == 0 and epoch > 99:
            plt.plot(global_loss_g)
            plt.savefig(str(epoch) + " epoch_g.jpg")
            plt.plot(global_loss_d)
            plt.savefig(str(epoch) + " epoch_d.jpg")
    
    plt.plot(global_loss_g)
    plt.savefig("final loss_g.jpg")
    plt.plot(global_loss_d)
    plt.savefig("final loss_d.jpg")
        
        

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
#     model = net.Net(params).cuda() if params.cuda else net.Net(params)
    netG = net.Generator(4).cuda(cuda_id)
    netD = net.Discriminator().cuda(cuda_id)
    
    optimG = optim.Adam(netG.parameters(),lr=params.learning_rate)
    optimD = optim.Adam(netD.parameters(),lr=params.learning_rate)
    if args.optim == 'sgd':
        optimG = optim.SGD(netG.parameters(),lr=params.learning_rate,momentum=0.9)
        optimD = optim.SGD(netD.parameters(),lr=params.learning_rate,momentum=0.9)

    # fetch loss function and metrics
#     loss_fn = net.loss_fn
    loss_fn = None
    if net == GAN or net == CGAN or net == CGAN_finetune:
        loss_fn = GeneratorLoss().cuda(cuda_id)
    elif net == GAN_adv:
        loss_fn = GeneratorLoss_adv().cuda(cuda_id)
    elif net == GAN_ssim:
        loss_fn = GeneratorLoss_ssim().cuda(cuda_id)
    elif net == GAN_notv:
        loss_fn = GeneratorLoss_notv().cuda(cuda_id)
    
    metrics = net.metrics
    
    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(netG, netD, train_dl, val_dl, optimG, optimD, loss_fn, metrics, params, args.model_dir,
                       args.restore_file,cuda_id=cuda_id)

    print("finish training and evaluating!")
    