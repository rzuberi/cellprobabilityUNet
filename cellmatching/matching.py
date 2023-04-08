from import_images import import_images_from_path, import_npy_files
import matplotlib.pyplot as plt
import torch
from u_net import UNet
import numpy as np
import time
import cv2
from multiprocessing import Pool


def get_enc_feats(encoder,images):
    images = np.array(images)

    images = torch.from_numpy(images.astype('float32'))
    images = images.unsqueeze(1)

    enc_feats = encoder(images)[2]

    enc_feats = enc_feats.detach().numpy()

    return enc_feats

def process_mask_value(mask_value):
    masked_enc_feat_1 = []
    for channel in range(enc_feat_1.shape[1]):
        interpolated_enc_feat = cv2.resize(enc_feat_1[0][channel], dsize=(1080, 1080), interpolation=cv2.INTER_CUBIC)
        values = np.where(masks[0] == mask_value, interpolated_enc_feat, 0)
        masked_enc_feat_1.append(values)
    return masked_enc_feat_1

if __name__ == '__main__':
    images = import_images_from_path('cellmatching/data/')
    
    #import the masks
    masks = import_npy_files('cellmatching/masks/')

    #import the model dicts
    #put the models in U-Nets
    cellprob_model = UNet()
    cellprob_model.load_state_dict(torch.load('cellmatching\model_dicts\cell-probability-map-U-Net-best.pt'))
    
    #get the encoders of each model
    cellprob_encoder = cellprob_model.encoder

    #get the encodings of the images
    enc_feat_1 = get_enc_feats(cellprob_encoder,images)

    #get the encodings of each cell
    #loop through the values in the masks starting from value 0
    #mask out the encodings on each channel and save them in a list
    masked_enc_feat_1_list = []
    for mask_value in range(np.max(masks[0])): #looping through each cell mask in the first image mask
        for channel in range(enc_feat_1.shape[1]): #looping through the 64 channels of the encoding of the first image
            #get the values in the masks that are equal to the mask_value
            interpolated_enc_feat = cv2.resize(enc_feat_1[0][channel], dsize=(1080, 1080), interpolation=cv2.INTER_CUBIC)
            values = np.where(masks[0] == mask_value, interpolated_enc_feat, 0)
            masked_enc_feat_1_list.append(values)
        print(mask_value)
    print(len(masked_enc_feat_1_list))


        #mask out the encodings
        #mask = np.where(masks[0] == mask_value, enc_feat_1[0], 0)
        #save the masked encodings in a list
        #masked_enc_feat_1_list.append(masked_enc_feat_1)
    #print(masked_enc_feat_1_list)

    #match them