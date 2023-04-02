from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision

import numpy as np
import matplotlib.pyplot as plt

# Define a custom dataset
class CustomDataset(Dataset):
    def __init__(self, file_list, label_list, transform=None):
        self.file_list = file_list
        self.label_list = label_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        image = Image.open(self.file_list[index])
        label = self.label_list[index]

        if self.transform:
            image = self.transform(image)

        return image, label

def get_data_loader(files, labels, transforms, batch_size=32, shuffle=True):
    """
    It takes a list of files and labels, applies a set of transforms to each file, and returns a data
    loader that can be used to iterate over the transformed data
    
    :param files: list of paths to the images
    :param labels: a list of labels for each image
    :param transforms: This is a list of transformations that will be applied to the images
    :param batch_size: The number of images to be passed through the network at once, defaults to 32
    (optional)
    :param shuffle: Whether to shuffle the data or keep it in the same order, defaults to True
    (optional)
    :return: A dataloader object
    """
    dataset_data = CustomDataset(files, labels, transform=transforms)
    return DataLoader(dataset_data, batch_size=batch_size, shuffle=shuffle)

def image_plotter(image_tensor, title=None, transform_color=True, show_axis=True, show_grid=False, figsize=(20, 20)):
    """
    It takes in a tensor, transforms it to a numpy array, transposes it, transforms it back to a tensor,
    and then plots it
    
    :param image_tensor: The image tensor to be plotted
    :param title: The title of the plot
    :param transform_color: If True, the image will be transformed from the range [-1, 1] to [0, 1],
    defaults to True (optional)
    :param show_axis: If True, the axis will be shown. If False, the axis will be hidden, defaults to
    True (optional)
    :param show_grid: Whether to show the grid or not, defaults to False (optional)
    :param figsize: The size of the figure to be plotted
    """
    image_tensor = image_tensor.numpy().transpose((1, 2, 0))
    
    if transform_color:
        mean = np.array(image_tensor)
        std = np.array(image_tensor)
        image_tensor = std * image_tensor + mean
        image_tensor = np.clip(image_tensor, 0, 1)
    else:
        image_tensor = image_tensor / 2 + 0.5

    plt.figure(figsize=figsize)
    plt.axis(show_axis)
    plt.grid(show_grid)

    plt.imshow(image_tensor)
    
    if title is not None:
        plt.title(title)
        
    plt.pause(0.001)

def plot_image_gallery(images, 
                       title=None, 
                       transform_color=True,
                       show_axis=True,
                       show_grid=False,
                       nrow=5,
                       normalize = False,
                       padding = 0,
                       figsize=(20, 20)):
    """
    It takes a list of images and plots them in a grid.
    
    :param images: a list of images to be plotted
    :param title: Title of the plot
    :param transform_color: If True, the image will be transformed to RGB. If False, the image will be
    transformed to grayscale, defaults to True (optional)
    :param show_axis: Whether to show the axis or not, defaults to True (optional)
    :param show_grid: Whether to show the grid or not, defaults to False (optional)
    :param nrow: number of images per row, defaults to 5 (optional)
    :param normalize: If True, shift the image to the range (0, 1), by the min and max values specified
    by range. Default: False, defaults to False (optional)
    :param padding: number of pixels to pad between images, defaults to 0 (optional)
    :param figsize: the size of the figure
    """
    image_tensor = torchvision.utils.make_grid(images, nrow=nrow, normalize = normalize, padding = padding)
    image_plotter(title=title,
                  transform_color=transform_color,
                  show_axis=show_axis, 
                  show_grid=show_grid, 
                  figsize=figsize,
                  image_tensor=image_tensor)
    