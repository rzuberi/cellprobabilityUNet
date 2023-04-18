from os import listdir
from os.path import isfile, join
import re
import numpy as np
from import_images import import_images_from_path
from cellmask_model import CellMaskModel
import os
import matplotlib.pyplot as plt
import torch
import cv2
import time
import pickle

def get_encodings(images, model, encoder_type):
    #get the encoder of the model
    if encoder_type == 'cp':
        encoder = model.unet_cp.encoder
    elif encoder_type == 'mask':
        encoder = model.unet_mask.encoder
    #torch.from_numpy(encFeats[0]).expand(0).expand(0
    encFeats = encoder(torch.from_numpy(images).unsqueeze(1).type(torch.float32))[2].detach().numpy()

    
    encFeats_interpolated = []
    for i in range(len(encFeats)):
        encFeats_per_channel = []
        for j in range(len(encFeats[i])):
            encFeats_per_channel.append(cv2.resize(encFeats[i][j], (1080,1080), interpolation=cv2.INTER_CUBIC))
        encFeats_interpolated.append(encFeats_per_channel)
    encFeats = np.array(encFeats_interpolated)
    return encFeats   

def get_encFeats_per_cell_per_mask(instance_masks, encFeats):
    encFeats_per_cell_per_mask = []
    for i in range(len(instance_masks)):
        instance_mask = instance_masks[i]
        encoding_features = encFeats[i]
        encoding_features_per_cell_2 = []
        for mask_val in range(1, np.max(instance_mask)+1):
            mask = instance_mask == mask_val
            masked_encFeats = encoding_features[:, mask]
            encoding_features_per_cell_2.append(masked_encFeats)
        encFeats_per_cell_per_mask.append(encoding_features_per_cell_2)

    encFeats_per_cell_per_mask = np.array(encFeats_per_cell_per_mask)
    return encFeats_per_cell_per_mask

def get_cos_sims_per_cell_per_mask(encFeats_per_cell_per_mask):
    cos_sims_for_each_cell = []
    for cell_encFeats in encFeats_per_cell_per_mask[0]:
        cos_sims_for_cell = []
        arr1 = cell_encFeats.flatten()
        for cell_encFeats2 in encFeats_per_cell_per_mask[1]:
            arr2 = cell_encFeats2.flatten()
            #if the length difference between arr1 and arr2 is over 25%, cos_sim == 0
            if abs(arr1.shape[0] - arr2.shape[0]) > 0.1 * max(arr1.shape[0], arr2.shape[0]):
                cos_sims_for_cell.append(0)
                continue

            #get the mean of each array
            #arr1_mean = np.mean(arr1)
            #arr2_mean = np.mean(arr2)
            #cos_sims_for_cell.append(abs(arr1_mean - arr2_mean)) #append the absolute value of the difference between the two means
            
        
            #pad the shorter array with the mean of the array
            if arr1.shape[0] > arr2.shape[0]:
                pad_by = arr1.shape[0]-arr2.shape[0]
                arr2 = np.pad(arr2, (0, pad_by), 'constant')
                #arr2 = np.pad(arr2, (0, arr1.shape[0] - arr2.shape[0]), 'mean')
                #arr1 = arr1[:arr2.shape[0]]
            elif arr1.shape[0] < arr2.shape[0]:
                arr2 = arr2[:arr1.shape[0]]
                #arr1 = np.pad(arr1, (0, arr2.shape[0] - arr1.shape[0]), 'mean')
                #arr2 = arr2[:arr1.shape[0]]
            cos_sim = np.dot(arr1, arr2) / (np.linalg.norm(arr1) * np.linalg.norm(arr2))
            cos_sims_for_cell.append(cos_sim)

        cos_sims_for_each_cell.append(cos_sims_for_cell)

    cos_sims_for_each_cell = np.array(cos_sims_for_each_cell)

    lst = cos_sims_for_each_cell.tolist()
    return lst

def get_pairs(cos_sims_lst):
    pairs = []
    length = min(len(cos_sims_lst),len(cos_sims_lst[0]))
    for i in range(length):
        cos_sims_lst = np.array(cos_sims_lst)
        max_index = np.argmax(cos_sims_lst)
        #print(max_index)
        row, col = np.unravel_index(max_index, cos_sims_lst.shape)
        if row not in [pair[0] for pair in pairs] and col not in [pair[1] for pair in pairs]: #to not get duplicate pairs
            pairs.append((row,col))
        cos_sims_lst[:,col] = 0
        cos_sims_lst[row,:] = 0
    return pairs

def merge(arr,arr1):
    result = []
    for i in arr:
        for j in arr1:
            if i[-1] == j[0]:
                res = [num for num in i]
                res.append(j[1])
                result.append(res)
    result = np.array(result)
    return result

def merge_list(list_of_pairs):
    matches = list_of_pairs[0]
    for i in range(len(list_of_pairs)-1):
        matches = merge(matches,list_of_pairs[i+1])
    matches = np.array(matches)
    return np.squeeze(matches)

def display_one_pair_list(pairs, cell_centers, images):
    for pair in pairs:
        first_cell_index = pair[0]
        second_cell_index = pair[1]

        first_cell_center = (round(cell_centers[0][first_cell_index][0]),round(cell_centers[0][first_cell_index][1]))
        second_cell_center = (round(cell_centers[1][second_cell_index][0]),round(cell_centers[1][second_cell_index][1]))

        print(first_cell_center, second_cell_center)

        crop = images[0][first_cell_center[0]-25:first_cell_center[0]+25, first_cell_center[1]-25:first_cell_center[1]+25]
        crop2 = images[1][second_cell_center[0]-25:second_cell_center[0]+25, second_cell_center[1]-25:second_cell_center[1]+25]

        plt.subplot(1,2,1)
        plt.imshow(crop)
        plt.subplot(1,2,2)
        plt.imshow(crop2)
        plt.show()

def display_match(cell_indexes, cell_centers, images, size_of_crop=50):
    size_of_crop = round(size_of_crop/2)

    for i in range(len(cell_indexes)):
        cell_index = cell_indexes[i]
        cell_center = (round(cell_centers[i][cell_index][0]),round(cell_centers[i][cell_index][1]))
        crop = images[i][cell_center[0]-size_of_crop:cell_center[0]+size_of_crop, cell_center[1]-size_of_crop:cell_center[1]+size_of_crop]

        plt.subplot(1,len(cell_indexes),i+1)
        plt.imshow(crop)
    plt.show()

def display_matches(list_of_cell_indexes, cell_centers, images, size_of_crop=50):
    size_of_crop = round(size_of_crop/2)

    crops = []
    for j in range(len(list_of_cell_indexes)):
        cell_indexes = list_of_cell_indexes[j]
        for i in range(len(cell_indexes)):
            cell_index = cell_indexes[i]
            cell_center = (round(cell_centers[i][cell_index][0]),round(cell_centers[i][cell_index][1]))
            crop = images[i][cell_center[0]-size_of_crop:cell_center[0]+size_of_crop, cell_center[1]-size_of_crop:cell_center[1]+size_of_crop]
            crops.append(crop)

    for i in range(len(crops)):
        plt.subplot(len(list_of_cell_indexes),len(list_of_cell_indexes[0]),i+1)
        plt.axis('off')
        plt.imshow(crops[i])
    plt.show()

if __name__ == '__main__':
    #import the cell_centers2.txt file without numpy
    with open("cell_centers", "rb") as fp:   # Unpickling
        cell_centers = pickle.load(fp)
    cell_centers = [[(item[0], item[1]) for item in arr] for arr in cell_centers]

    #import the images#
    images = np.array(import_images_from_path('cellmask/data/',num_imgs=3,normalisation=True))

    model = CellMaskModel()
    model.import_model(os.getcwd() + '/cellmask/saved_weights/cp_model', os.getcwd() + '/cellmask/saved_weights/mask_model')
    #cps, masks, instance_masks = model.eval(images) #Making predictions

    #Let's import the ground truth instance masks which are npy files
    instance_masks_path = str(os.getcwd()) + '\\cellmask\\data_for_cellpose\\'
    onlyfiles = [f for f in listdir(instance_masks_path) if isfile(join(instance_masks_path, f)) and f.endswith('.npy')]
    onlyfiles.sort(key=lambda f: int(re.sub('\D', '', f))) #sort the files in order
    gt_instance_masks = [np.load(instance_masks_path + onlyfiles[i], allow_pickle=True).item()['masks'] for i in range(len(onlyfiles))]
    gt_instance_masks = gt_instance_masks[:3]

    #start = time.time()
    encodings_cp = get_encodings(images, model, encoder_type='cp')
    encodings_mk = get_encodings(images, model, encoder_type='mask')

    #I want to get the encodings from both encoders
    #Then I want to average them together

    encFeats_per_cell_per_mask_cp = get_encFeats_per_cell_per_mask(gt_instance_masks, encodings_cp)
    #print(encFeats_per_cell_per_mask_cp[0])
    encFeats_per_cell_per_mask_mk = get_encFeats_per_cell_per_mask(gt_instance_masks, encodings_mk)

    #print(encFeats_per_cell_per_mask_mk.shape)
    cos_sims_per_cell_masks_encFeats_cp = get_cos_sims_per_cell_per_mask(encFeats_per_cell_per_mask_cp)
    cos_sims_per_cell_masks_encFeats_mk = get_cos_sims_per_cell_per_mask(encFeats_per_cell_per_mask_mk)
    cos_sims_per_cell_masks_encFeats_mean = np.mean([cos_sims_per_cell_masks_encFeats_cp, cos_sims_per_cell_masks_encFeats_mk], axis=0)
    print(cos_sims_per_cell_masks_encFeats_cp[1])
    pairs_1 = get_pairs(cos_sims_per_cell_masks_encFeats_cp)

    #cos_sims_per_cell_masks_encFeats_cp = get_cos_sims_per_cell_per_mask(encFeats_per_cell_per_mask_cp[1:3])
    #cos_sims_per_cell_masks_encFeats_mk = get_cos_sims_per_cell_per_mask(encFeats_per_cell_per_mask_mk[1:3])
    #cos_sims_per_cell_masks_encFeats_mean = np.mean([cos_sims_per_cell_masks_encFeats_cp, cos_sims_per_cell_masks_encFeats_mk], axis=0)
    #pairs_2 = get_pairs(cos_sims_per_cell_masks_encFeats_cp)

    #merged_list = merge_list([pairs_1,pairs_2])
    #print(merged_list)
    print(pairs_1)
    for i in range(0,len(pairs_1)-5,5):
        print(pairs_1[i:i+5])
        display_matches(pairs_1[i:i+5], cell_centers, images, size_of_crop=50)

    #print('time taken: ', time.time()-start)