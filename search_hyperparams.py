"""Peform hyperparemeters search"""

import argparse
import os
from subprocess import check_call
import sys

import utils


PYTHON = sys.executable
parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir', default='experiments/learning_rate',
                    help='Directory containing params.json')
parser.add_argument('--data_dir', default='data/64x64_SIGNS', help="Directory containing the dataset")
parser.add_argument('--model', default='srcnn', help="filename of the model")
parser.add_argument('--model_type', default='cnn', help="type of the model")


def launch_training_job(parent_dir, data_dir, job_name, params, model, model_type):
    """Launch training of the model with a set of hyperparameters in parent_dir/job_name

    Args:
        model_dir: (string) directory containing config, weights and log
        data_dir: (string) directory containing the dataset
        params: (dict) containing hyperparameters
    """
    # Create a new folder in parent_dir with unique_name "job_name"
    model_dir = os.path.join(parent_dir, job_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Write parameters in json file
    json_path = os.path.join(model_dir, 'params.json')
    params.save(json_path)

    # Launch training with this config
    cmd = None
    if model_type == "cnn":
        cmd = "{python} train_cnn.py --model_dir {model_dir} --data_dir {data_dir} --model {model} --cuda cuda0 --optim adam".format(python=PYTHON, model_dir=model_dir, data_dir=data_dir, model=model)    
    elif model_type == "gan":
        cmd = "{python} train_gan.py --model_dir {model_dir} --data_dir {data_dir} --model {model} --cuda cuda0 --optim adam".format(python=PYTHON, model_dir=model_dir, data_dir=data_dir, model=model)
        
        
    print(cmd)
    check_call(cmd, shell=True)


if __name__ == "__main__":
    # Load the "reference" parameters from parent_dir json file
    args = parser.parse_args()
    json_path = os.path.join(args.parent_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # Perform hypersearch over one parameter
    learning_rates = [1e-3, 5e-4, 1e-4]
    
    
    for learning_rate in learning_rates:
        # Modify the relevant parameter in params
        params.learning_rate = learning_rate
        print("get learning rate")
        # Launch job (name has to be unique)
        job_name = "learning_rate_{}".format(learning_rate)
        launch_training_job(args.parent_dir, args.data_dir, job_name, params, args.model, args.model_type)
        print("finished one")