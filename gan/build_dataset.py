"""Split the SIGNS dataset into train/val/test and resize images to 64x64.

The SIGNS dataset comes into the following format:
    train_signs/
        0_IMG_5864.jpg
        ...
    test_signs/
        0_IMG_5942.jpg
        ...

Original images have size (3024, 3024).
Resizing to (64, 64) reduces the dataset size from 1.16 GB to 4.7 MB, and loading smaller images
makes training faster.

We already have a test set created, so we only need to split "train_signs" into train and val sets.
Because we don't have a lot of images and we want that the statistics on the val set be as
representative as possible, we'll take 20% of "train_signs" as val set.
"""

import argparse
import random
import os

from PIL import Image
from tqdm import tqdm

from skimage import io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
from skimage import util

SIZE = 256

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/faces', help="Directory with the SIGNS dataset")
parser.add_argument('--output_dir', default='data/srcnn_FACES', help="Where to write the new data")


def resize_and_save(filename, output_dir, size=SIZE):
    """Resize the image contained in `filename` and save it to the `output_dir`"""
    #image = Image.open(filename)
    # Use bilinear interpolation instead of the default "nearest neighbor" method
    #image = image.resize((size, size), Image.BILINEAR)
    #image.save(os.path.join(output_dir, filename.split('/')[-1]))
    image = io.imread(filename)
    cropped = image[:,20:198] # 178*218 -> 178*178
    image_resized = resize(cropped, (64, 64)) # 21*21
    io.imsave(os.path.join(output_dir, filename.split('/')[-1]), image_resized)
    
def blur_and_save(filename, output_dir):
    """Resize the image contained in `filename` and save it to the `output_dir`"""
    #image = Image.open(filename)
    # Use bilinear interpolation instead of the default "nearest neighbor" method
    #image = image.resize((size, size), Image.BILINEAR)
    
    image = io.imread(filename)
    cropped = image[:,20:198] # 178*218 -> 178*178
    image_resized = resize(cropped, (64, 64)) # resize to 64 * 64
    #high_res = rescale(image_resized, 1.0 / 2.0) # rescale to useless.
    image_downscaled = downscale_local_mean(image_resized, (2, 2, 1)) # downscale to 32*32 (blur)
    
    # srcnn
#     image_upsized = resize(image_downscaled,(76,76)) # 32*32 -> 76*76
       
    # fsrcnn
    image_upsized = image_downscaled

    io.imsave(os.path.join(output_dir, filename.split('/')[-1]), image_upsized)

if __name__ == '__main__':
    args = parser.parse_args()

    assert os.path.isdir(args.data_dir), "Couldn't find the dataset at {}".format(args.data_dir)

    # Define the data directories
    train_data_dir = os.path.join(args.data_dir, 'train_faces')
    test_data_dir = os.path.join(args.data_dir, 'test_faces')

    # Get the filenames in each directory (train and test)
    filenames = os.listdir(train_data_dir)
    filenames = [os.path.join(train_data_dir, f) for f in filenames if f.endswith('.jpg')]

    test_filenames = os.listdir(test_data_dir)
    test_filenames = [os.path.join(test_data_dir, f) for f in test_filenames if f.endswith('.jpg')]

    # Split the images in 'train_signs' into 80% train and 20% val
    # Make sure to always shuffle with a fixed seed so that the split is reproducible
    random.seed(230)
    filenames.sort()
    random.shuffle(filenames)

    split = int(0.95 * len(filenames))
    train_filenames = filenames[:split]
    val_filenames = filenames[split:]

    filenames = {'train': train_filenames,
                 'val': val_filenames,
                 'test': test_filenames}
    
    
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    else:
        print("Warning: output dir {} already exists".format(args.output_dir))
 
    

    # Preprocess train, val and test
    for split in ['train', 'val', 'test']:
        # clear image
        output_dir_split = os.path.join(args.output_dir, '{}_faces'.format(split))
        # blur image
        output_dir_split_blur = os.path.join(args.output_dir, '{}_faces_blur'.format(split))
 
        # clean image
        if not os.path.exists(output_dir_split):
            os.mkdir(output_dir_split)
        else:
            print("Warning: dir {} already exists".format(output_dir_split))
        # blur image
        if not os.path.exists(output_dir_split_blur):
            os.mkdir(output_dir_split_blur)
        else:
            print("Warning: dir {} already exists".format(output_dir_split_blur))
            
        print("Processing {} data, saving preprocessed data to {}".format(split, output_dir_split))
        # clean image
        for filename in tqdm(filenames[split]):
            resize_and_save(filename, output_dir_split, size=SIZE)
        #blur image
        for filename in tqdm(filenames[split]):
            blur_and_save(filename, output_dir_split_blur)

    print("Done building dataset")
    