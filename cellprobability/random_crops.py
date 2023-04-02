#here we use PyTorch to get random crops of images

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np
import matplotlib.pyplot as plt
from import_images import import_images_from_path
from cellpose_data import get_cellpose_probability_maps

def get_one_random_crop(image,cellprob):

    image_torch = torch.from_numpy(image).float().to('cuda:0')
    cellprob_torch = torch.from_numpy(cellprob).to('cuda:0')

    i, j, h, w = transforms.RandomCrop.get_params(image_torch, output_size=(128, 128))

    image_crop = TF.crop(image_torch, i, j, h, w)
    cellprob_crop = TF.crop(cellprob_torch, i, j, h, w)

    return image_crop, cellprob_crop

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