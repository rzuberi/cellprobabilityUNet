#import the images from the data_for_cellpose folder

#Imports
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from skimage import measure
from statistics import mean
import PySimpleGUI as sg
import numpy as np

images_path = str(os.getcwd()) + '\\data_for_cellpose\\'
onlyfiles = [f for f in listdir(images_path) if isfile(join(images_path, f)) and f.endswith('.tiff')]
images = [np.squeeze(tifffile.imread(images_path +  onlyfiles[i])) for i in range(len(onlyfiles))]

instance_masks_path = str(os.getcwd()) + '\\data_for_cellpose\\'
onlyfiles = [f for f in listdir(images_path) if isfile(join(images_path, f)) and f.endswith('.npy')]
gt_instance_masks = [np.load(instance_masks_path + onlyfiles[i], allow_pickle=True).item()['masks'] for i in range(len(onlyfiles))]

binary_gt_instance_masks = [np.where(gt_instance_masks[i] > 0, 1, 0) for i in range(len(gt_instance_masks))]

def instance_seg_alg(binary_mask):
    return measure.label(binary_mask, background=0,connectivity=1)

instance_masks = [instance_seg_alg(binary_gt_instance_masks[i]) for i in range(len(binary_gt_instance_masks))]

num_cells = np.array([np.max(instance_masks[i]) for i in range(len(instance_masks))])
num_gt_cells = np.array([np.max(gt_instance_masks[i]) for i in range(len(gt_instance_masks))])

crop_size = 108
n_crops = images[0].shape[0] // crop_size
crops = []
for image in images:
    for i in range(n_crops):
        for j in range(n_crops):
            crop = image[i*crop_size:(i+1)*crop_size, j*crop_size:(j+1)*crop_size]
            crops.append(crop)
print(len(crops))


def array_to_data(array):
    im = Image.fromarray(array)
    with BytesIO() as output:
        im.save(output, format="PNG")
        data = output.getvalue()
    return data

import PySimpleGUI as sg
import numpy as np
import io
from PIL import Image
from io import BytesIO


# Define the layout of the GUI
layout = [
    [sg.Image(data='', key='-IMAGE-')],
    [sg.Button('Keep'), sg.Button('Discard')],
    [sg.Text('Kept: 0   Discarded: 0', key='-COUNTER-')]
]

# Create the window
window = sg.Window('Image Viewer', layout, finalize=True)

# Define the lists for keeping and discarding images
keep_list = []
discard_list = []

# Loop over the images
images = crops  # replace with your numpy arrays
for image in images:
    # Convert the numpy array to bytes and update the image in the GUI
    #bio = io.BytesIO()
    #np.save(bio, image)
    window['-IMAGE-'].update(data=array_to_data(image))

    # Wait for a button press
    event, values = window.read()

    # Handle the button press
    if event == 'Keep':
        keep_list.append(image)
    elif event == 'Discard':
        discard_list.append(image)

    # Update the counters in the GUI
    window['-COUNTER-'].update('Kept: {}   Discarded: {}'.format(len(keep_list), len(discard_list)))

# Close the window
window.close()
