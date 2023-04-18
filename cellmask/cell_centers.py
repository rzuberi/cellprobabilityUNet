import os
from os import listdir
from os.path import isfile, join
import re
import numpy as np
import matplotlib.pyplot as plt
import ast
import pickle

instance_masks_path = str(os.getcwd()) + '\\cellmask\\data_for_cellpose\\'
onlyfiles = [f for f in listdir(instance_masks_path) if isfile(join(instance_masks_path, f)) and f.endswith('.npy')]
onlyfiles.sort(key=lambda f: int(re.sub('\D', '', f))) #sort the files in order
gt_instance_masks = [np.load(instance_masks_path + onlyfiles[i], allow_pickle=True).item()['masks'] for i in range(len(onlyfiles))]

#get the centers of each unique value in first_gt except 0
found_centers = []
for mask in gt_instance_masks:
    centers = np.array([(np.where(mask == i)[0].mean(), np.where(mask == i)[1].mean()) for i in np.unique(mask) if i != 0])
    found_centers.append(centers)

with open("cell_centers", "wb") as fp:   #Pickling
    pickle.dump(found_centers, fp)

#with open("cell_centers", "rb") as fp:   # Unpickling
#    cell_centers = pickle.load(fp)
