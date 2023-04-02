#Here we take torch images and we return augmentations
#For now we only return rotations

import torch
import matplotlib.pyplot as plt

from import_images import import_images_from_path
from cellpose_data import get_cellpose_probability_maps
from random_crops import get_random_crops_from_multiple_images

def rotate_torch_image(image,angle):
    if angle == 90:
        return torch.rot90(image,3)
    elif angle == 180:
        return torch.rot90(image,2)
    elif angle == 270:
        return torch.rot90(image,1)
    return torch.rot90(image,1)

def rotate_multiple_images(images,angles):
    #Take the images and rotate them by the angles
    #Add to a list
    rotated_images = []
    for image in images:
        rotated_images.append(image)
        for angle in angles:
            rotated_images.append(rotate_torch_image(image,angle))
    return rotated_images

def rotate_images_and_cellprobs(images,cellprobs,angles):
    rotated_images = rotate_multiple_images(images,angles)
    rotated_cellprobs = rotate_multiple_images(cellprobs,angles)
    return rotated_images, rotated_cellprobs

def rotate_images_and_cellprobs_return_merged(images,cellprobs,angles):
    rotated_images, rotated_cellprobs = rotate_images_and_cellprobs(images,cellprobs,angles)
    return rotated_images, rotated_cellprobs

if __name__ == '__main__':
    images = import_images_from_path('data/',num_imgs=1,normalisation=True)
    cellprobs = get_cellpose_probability_maps(images)
    images, cellprobs = get_random_crops_from_multiple_images(images,cellprobs,num_crops=10)
    rotated_images, rotated_cellprobs = rotate_images_and_cellprobs_return_merged(images,cellprobs,angles=[90,180,270])

    for i in range(10):
        plt.subplot(2,10,i+1)
        plt.imshow(rotated_images[i].cpu().numpy())
        plt.axis('off')
        plt.subplot(2,10,i+11)
        plt.imshow(rotated_cellprobs[i].cpu().numpy())
        plt.axis('off')
    plt.show()