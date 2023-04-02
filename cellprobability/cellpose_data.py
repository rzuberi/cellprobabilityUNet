#Here we import cellpose and make a function to give it some images and it returns the cellprob maps
#make cellpose use the gpu

import cellpose
from cellpose import models, core
import numpy as np
import matplotlib.pyplot as plt

from import_images import import_images_from_path

def get_cellpose_probability_maps(images):
    # make a prediction on the data with cellpose
    model = models.Cellpose(gpu=core.use_gpu(), model_type='nuclei')
    masks, flows, styles, diams = model.eval(images)

    cellprobs = []

    for flows_per_img in flows:
        cellprobs.append(flows_per_img[2])

    cellprobs = np.array(cellprobs)

    return cellprobs

if __name__ == '__main__':
    images = import_images_from_path('data/',num_imgs=3,normalisation=True)
    cellprobs = get_cellpose_probability_maps(images)#
    for i in range(len(cellprobs)):
        plt.subplot(1,3,i+1)
        plt.imshow(cellprobs[i])
    plt.show()
    print(cellprobs.shape)