#here we use PyTorch to get random crops of images

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np
import matplotlib.pyplot as plt
from import_images import import_images_from_path
from cellpose_data import get_cellpose_probability_maps
import random

def get_one_random_crop(x, y, crop_size=(128,128)):
    assert x.shape[0] == y.shape[0]
    assert x.shape[1] == y.shape[1]
    h, w = x.shape
    rangew = (w - crop_size[0]) // 2 if w>crop_size[0] else 0
    rangeh = (h - crop_size[1]) // 2 if h>crop_size[1] else 0
    offsetw = 0 if rangew == 0 else np.random.randint(rangew)
    offseth = 0 if rangeh == 0 else np.random.randint(rangeh)
    cropped_x = x[offseth:offseth+crop_size[0], offsetw:offsetw+crop_size[1]]
    cropped_y = y[offseth:offseth+crop_size[0], offsetw:offsetw+crop_size[1]]
    #cropped_y = cropped_y[:, :, ~np.all(cropped_y==0, axis=(0,1))]
    if cropped_y.shape[-1] == 0:
        return x, y
    else:
        return cropped_x, cropped_y

def get_random_crops_from_one_image(image,cellprob,num_crops=10):
    image_crops = []
    cellprob_crops = []
    for i in range(num_crops):
        image_crop,cellprob_crop = get_one_random_crop(image,cellprob)
        image_crops.append(image_crop)
        cellprob_crops.append(cellprob_crop)
    return image_crops, cellprob_crops

def get_random_crops_from_multiple_images(images,cellprobs,num_crops=10):
    image_crops_list = []
    cellprob_crops_list = []

    for i in range(len(images)):
        image = images[i]
        cellprob = cellprobs[i]
        image_crops,cellprob_crops = get_random_crops_from_one_image(image,cellprob,num_crops=num_crops)
        image_crops_list.append(image_crops)
        cellprob_crops_list.append(cellprob_crops)

    #merge the lists inside image_crops_lists
    image_crops_lists_merged = [item for sublist in image_crops_list for item in sublist]
    cellprob_crops_list_merged = [item for sublist in cellprob_crops_list for item in sublist]
    return image_crops_lists_merged, cellprob_crops_list_merged

if __name__ == '__main__':
    print(torch. __version__)
    print(torch.cuda.is_available())
    images = import_images_from_path('data/',num_imgs=1,normalisation=True)
    cellprobs = get_cellpose_probability_maps(images)
    random_crops = get_random_crops_from_multiple_images(images,cellprobs,num_crops=10)
    print(len(random_crops))
    print(len(random_crops[0]))