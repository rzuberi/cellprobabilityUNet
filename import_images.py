import tifffile
import numpy as np
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt

def import_images_from_path(images_path,normalisation=False,num_imgs=20,format='.tif'):
    onlyfiles = [f for f in listdir(images_path) if isfile(join(images_path, f))] #not ordered in the right way

    if num_imgs > len(onlyfiles):
        num_imgs = len(onlyfiles)

    images = [np.squeeze(tifffile.imread(images_path +  onlyfiles[i])) for i in range(num_imgs)]

    if normalisation == True:
        return [(image-np.min(image))/(np.max(image)-np.min(image)) for image in images]
    return images

if __name__ == '__main__':
    images = import_images_from_path('data/',normalisation=True)
    # get square root of number of images
    sqrtlen = lambda x: int(np.ceil(np.sqrt(len(x))))
    print(len(images))
    for i in range(len(images)):
        plt.subplot(4,5,i+1)
        plt.imshow(images[i])
    plt.show()