import random
import os
import numpy as np

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

# borrowed from http://pytorch.org/tutorials/advanced/neural_style_tutorial.html
# and http://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# define a training image loader that specifies transforms on images. See documentation for more details.

'''
train_transformer = transforms.Compose([
    transforms.Resize(64),  # resize the image to 64x64 (remove if images are already 64x64)
    transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
    transforms.ToTensor()])  # transform it into a torch tensor

# loader for evaluation, no horizontal flip
eval_transformer = transforms.Compose([
    transforms.Resize(64),  # resize the image to 64x64 (remove if images are already 64x64)
    transforms.ToTensor()])  # transform it into a torch tensor
'''

train_transformer = transforms.Compose([transforms.ToTensor()])
eval_transformer = transforms.Compose([transforms.ToTensor()])


class FACESDataset(Dataset):
    """
    A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
    """
    
    # Note that the first directory is train, the second directory is label
    def __init__(self, blur_data_dir, data_dir, transform):  ########################### Here add a new parameter "blur_data_dir"
        """
        Store the filenames of the jpgs to use. Specifies transforms to apply on images.
        Args:
            data_dir: (string) directory containing the dataset
            transform: (torchvision.transforms) transformation to apply on image
        """
        self.filenames = os.listdir(data_dir)  # label
        self.filenames = [os.path.join(data_dir, f) for f in self.filenames if f.endswith('.jpg')] # label
        
        ########################### Here copy the filenames of labels, labels == blur_filenames
        self.blur_filenames = os.listdir(blur_data_dir)  # train
        self.blur_filenames = [os.path.join(blur_data_dir, f) for f in self.blur_filenames if f.endswith('.jpg')]

        self.transform = transform

    def __len__(self):
        # return size of dataset
        return len(self.filenames)

    def __getitem__(self, idx):
        """
        Fetch index idx image and labels from dataset. Perform transforms on image.
        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]
        Returns:
            image: (Tensor) transformed blur image
            label: (Tensor) transformed original image
        """
        
        # At this step everything is ok
        label_image = Image.open(self.filenames[idx])  # PIL image, label
        train_image = Image.open(self.blur_filenames[idx])  # train

        label_image = self.transform(label_image)  # label
        train_image = self.transform(train_image)  # train
  
        return train_image, label_image
#         return train_image


# Note that the first directory is train(blur), the second directory is label(clear)
def fetch_dataloader(types, data_dir, params):  
    ########################### Here add a new parameter "blur_data_dir", which is train
    """
    Fetches the DataLoader object for each type in types from data_dir.
    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        data_dir: (string) directory containing the dataset
        params: (Params) hyperparameters
    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """
    dataloaders = {}
    for split in ['train', 'val', 'test']:
        if split in types:
            path_blur = os.path.join(data_dir, "{}_blur".format(split))
            path = os.path.join(data_dir,  "{}_clear".format(split))
            
            # use the train_transformer if training data, else use eval_transformer without random flip
            if split == 'train':
                dl = DataLoader(FACESDataset(path_blur, path, train_transformer), batch_size=params.batch_size, shuffle=True,
                                        num_workers=params.num_workers,
                                        pin_memory=params.cuda)
            else:
                dl = DataLoader(FACESDataset(path_blur, path, eval_transformer), batch_size=params.batch_size, shuffle=False,
                                num_workers=params.num_workers,
                                pin_memory=params.cuda)

            dataloaders[split] = dl

    return dataloaders