#here we get cellpose data but using a trained model we import

import cellpose
from cellpose import models, core
import numpy as np
import matplotlib.pyplot as plt

from import_images import import_images_from_path

#take a path
#return a cellpose model from the path
def import_cellpose_model(model_path):
    model = models.CellposeModel(gpu=core.use_gpu(), pretrained_model=model_path, model_type='nuclei')
    #model.load_model(model_path)
    return model

def get_cellpose_probability_maps_pre_trained(model,images):
    model = models.Cellpose(gpu=core.use_gpu(), model_type='nuclei')
    masks, flows, styles, diams = model.eval(images)

    cellprobs = []

    for flows_per_img in flows:
        cellprobs.append(flows_per_img[2])

    cellprobs = np.array(cellprobs)

    return cellprobs

if __name__ == '__main__':
    model_path = 'cellmask/models/CP_20230402_212503_3'
    model = import_cellpose_model(model_path)
    print(model)

    images = import_images_from_path('cellmask/data/',num_imgs=3,normalisation=True)
    cellprobs = get_cellpose_probability_maps_pre_trained(model,images)
    for i in range(len(cellprobs)):
        plt.subplot(1,3,i+1)
        plt.imshow(cellprobs[i])
    plt.show()
    print(cellprobs.shape)