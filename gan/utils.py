import json
import logging
import os
import shutil

import torch

class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)
            
    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


class RunningAverage():
    """A simple class that maintains the running average of a quantity
    
    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """
    def __init__(self):
        self.steps = 0
        self.total = 0
    
    def update(self, val):
        self.total += val
        self.steps += 1
    
    def __call__(self):
        return self.total/float(self.steps)
        
    
def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def save_checkpoint(state, is_best, checkpoint, flag):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'

    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    if flag == 'G':
        filepath = os.path.join(checkpoint, 'last_g.pth.tar')
        if not os.path.exists(checkpoint):
            print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
            os.mkdir(checkpoint)
        else:
            print("Checkpoint Directory exists! ")
        torch.save(state, filepath)
        if is_best:
            shutil.copyfile(filepath, os.path.join(checkpoint, 'best_g.pth.tar'))
    else:
        filepath = os.path.join(checkpoint, 'last_d.pth.tar')
        if not os.path.exists(checkpoint):
            print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
            os.mkdir(checkpoint)
        else:
            print("Checkpoint Directory exists! ")
        torch.save(state, filepath)
        if is_best:
            shutil.copyfile(filepath, os.path.join(checkpoint, 'best_d.pth.tar'))


def load_checkpoint(checkpoint, model, optimizer=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.

    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise("File doesn't exist {}".format(checkpoint))
    
    
    if checkpoint == "experiments/cgan_finetune_model/best_g.pth.tar":
        checkpoint = torch.load(checkpoint,map_location={'cuda:1':'cuda:0'})
        pretrained_dict = model.state_dict()
#         print(pretrained_dict)
#         for i in range(1,6):
#             pretrained_list.append("block"+str(i)+".0")
#             pretrained_list.append("block"+str(i)+".1")

        pretrained_weight = checkpoint['state_dict']
        model_dict = model.state_dict()
        model_wanted = {}
        for k, v in model_dict.items():
            if int(k[5]) > 6:
                pass
            else :
                model_wanted[k] = v
            
#         for key in pretrained_dict:
#             pretrained_weight[key] = checkpoint['state_dict'][key]
#         model.load_state_dict(pretrained_weight)
#         print("model state: ", pretrained_dict.keys())
# #         print("pretrained weight: ", checkpoint['state_dict'].keys())
#     else:
# #         checkpoint = torch.load(checkpoint,map_location={'cuda:1':'cuda:0'})
#         print("cpkt: ", checkpoint['state_dict'].keys())
        model_dict.update(model_wanted)
        model.load_state_dict(model_dict)
    
#     if optimizer:
#         optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint
