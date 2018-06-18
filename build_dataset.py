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

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='../img_align_celeba_test', help="Directory with the SIGNS dataset")
parser.add_argument('--output_dir', default='../data/cnn_faces', help="Where to write the new data")
parser.add_argument('--input_size', default='144', help="Where to write the new data")
parser.add_argument('--output_size', default='144', help="Where to write the new data")
parser.add_argument('--up_scale', default='4', help="Where to write the new data")

def crop_and_save(filename, output_dir, out_size):
    """crop the image contained in `filename` and save it to the `output_dir`"""
    image = io.imread(filename)
    vert_start = (218 - out_size) // 2
    vert_end = vert_start + out_size
    horiz_start = (178 - out_size) // 2
    horiz_end = horiz_start + out_size
    cropped = image[vert_start:vert_end, horiz_start:horiz_end] # 218*178 -> 144*144
    io.imsave(os.path.join(output_dir, filename.split('/')[-1]), cropped)
    
def blur_and_save(filename, output_dir, in_size, out_size, up_scale):
    """Blur the image contained in `filename` and save it to the `output_dir`"""
    image = io.imread(filename)
    vert_start = (218 - out_size) // 2
    vert_end = vert_start + out_size
    horiz_start = (178 - out_size) // 2
    horiz_end = horiz_start + out_size
    
    cropped = image[vert_start:vert_end, horiz_start:horiz_end] # 218*178 -> 144*144    
    image_resized = resize(cropped, (out_size // up_scale, out_size // up_scale)) # upscaling factor 4
    blur = resize(image_resized, (in_size, in_size)) # rescale back to 144 * 144
    io.imsave(os.path.join(output_dir, filename.split('/')[-1]), blur)

if __name__ == '__main__':
    args = parser.parse_args()

    assert os.path.isdir(args.data_dir), "Couldn't find the dataset at {}".format(args.data_dir)

    # get args
    data_dir = args.data_dir
    INPUT_SIZE = int(args.input_size)
    OUTPUT_SIZE = int(args.output_size)
    UP_SCALE = int(args.up_scale)
    
    # Get the filenames in data directory
    filenames = os.listdir(data_dir)
    filenames = [os.path.join(data_dir, f) for f in filenames if f.endswith('.jpg')]

    # Split the images into 98% train, 1% val, and 1% test
    # Make sure to always shuffle with a fixed seed so that the split is reproducible
    random.seed(230)
    filenames.sort()
    random.shuffle(filenames)

    split1 = int(0.98 * len(filenames))
    split2 = (len(filenames) - split1) // 2 + split1
    train_filenames = filenames[:split1]
    val_filenames = filenames[split1:split2]
    test_filenames = filenames[split2:]
    
    print("train", len(train_filenames))
    print("val", len(val_filenames))
    print("test", len(test_filenames))
    
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
        output_dir_split_clear = os.path.join(args.output_dir, '{}_clear'.format(split))
        # blur image
        output_dir_split_blur = os.path.join(args.output_dir, '{}_blur'.format(split))
 
        # clear image
        if not os.path.exists(output_dir_split_clear):
            os.mkdir(output_dir_split_clear)
        else:
            print("Warning: dir {} already exists".format(output_dir_split_clear))
        # blur image
        if not os.path.exists(output_dir_split_blur):
            os.mkdir(output_dir_split_blur)
        else:
            print("Warning: dir {} already exists".format(output_dir_split_blur))
            
        print("Processing {} data, saving to {} and {}".format(split, output_dir_split_clear, output_dir_split_blur))
        # clear image
        for filename in tqdm(filenames[split]):
            crop_and_save(filename, output_dir_split_clear, OUTPUT_SIZE)
        #blur image
        for filename in tqdm(filenames[split]):
            blur_and_save(filename, output_dir_split_blur, INPUT_SIZE, OUTPUT_SIZE, UP_SCALE)
    
    print("Done building dataset")
    