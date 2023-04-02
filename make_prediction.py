#Here we take a 1080x1080 image and split it into equal crops of 128x128
#We then make predictions on each of these crops with the U-Net
#Then we reconstruct the image from the predictions and give it back

import numpy as np
import matplotlib.pyplot as plt
from import_images import import_images_from_path

def split_image_into_crops(image,crop_size):
    #Take the image and split it into crops of size crop_size
    #Return a list of crops
    
    #first pad the image by 100px
    image = np.pad(image,((100,100),(100,100)),'constant')
    print(image.shape)

    crops = []
    for i in range(0,image.shape[0],crop_size):
        for j in range(0,image.shape[1],crop_size):
            crops.append(image[i:i+crop_size,j:j+crop_size])
    return crops

def reconstruct_image_from_crops(crops,crop_size):
    #Take the crops and reconstruct the image
    #Return the reconstructed image
    image = np.zeros((crop_size*int(np.sqrt(len(crops))),crop_size*int(np.sqrt(len(crops)))))
    for i in range(int(np.sqrt(len(crops)))):
        for j in range(int(np.sqrt(len(crops)))):
            image[i*crop_size:(i+1)*crop_size,j*crop_size:(j+1)*crop_size] = crops[i*int(np.sqrt(len(crops)))+j]
    #unpad the image by 100px
    image = image[100:-100,100:-100]
    return image

def make_prediction_on_crop(model,crop):
    #Take a crop and make a prediction
    #Return the prediction
    prediction = model.eval(crop)
    return prediction

def make_prediction(model,image):
    #Take an image and make a prediction
    #Return the prediction
    crops = split_image_into_crops(image,128)
    predictions = []
    for crop in crops:
        predictions.append(make_prediction_on_crop(model,crop))
    prediction = reconstruct_image_from_crops(predictions,128)
    
    return prediction

if __name__ == '__main__':
    image = import_images_from_path('data/',num_imgs=1,normalisation=True)[0]
    crops = split_image_into_crops(image,128)
    print(len(crops))
    print(crops[0].shape)
    image_reconstructed = reconstruct_image_from_crops(crops,128)
    plt.imshow(image)
    plt.show()
    plt.imshow(image_reconstructed)
    plt.show()

    for i in range(len(crops)):
        plt.subplot(10,10,i+1)
        plt.imshow(crops[i])
        plt.axis('off')
    plt.show()