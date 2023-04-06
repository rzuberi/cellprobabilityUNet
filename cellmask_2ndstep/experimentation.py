from os import listdir
from os.path import isfile, join
import tifffile
import numpy as np
from cellpose import models, core
import os
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.optim import Adam
import wandb
from torch.utils.data import Dataset, DataLoader
from u_net import UNet


def import_images_from_path(images_path,normalisation=False,num_imgs=20):
    onlyfiles = [f for f in listdir(images_path) if isfile(join(images_path, f))] #not ordered in the right way
    if num_imgs > len(onlyfiles):
        num_imgs = len(onlyfiles)
    images = [np.squeeze(tifffile.imread(images_path +  onlyfiles[i])) for i in range(num_imgs)]
    if normalisation == True:
        return [(image-np.min(image))/(np.max(image)-np.min(image)) for image in images]
    return images

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

def get_one_random_crop(x, y, crop_size=(128,128)):
    assert x.shape[0] == y.shape[0]
    assert x.shape[1] == y.shape[1]
    h, w = x.shape
    rangew = (w - crop_size[0]) // 2 if w>crop_size[0] else 0
    rangeh = (h - crop_size[1]) // 2 if h>crop_size[1] else 0
    offsetw = 0 if rangew == 0 else np.random.randint(rangew)
    offseth = 0 if rangeh == 0 else np.random.randint(rangeh)
    cropped_x = x[offseth:offseth+crop_size[0], offsetw:offsetw+crop_size[1]]
    cropped_y = y[offseth:offseth+crop_size[0], offsetw:offsetw+crop_size[1]]
    #cropped_y = cropped_y[:, :, ~np.all(cropped_y==0, axis=(0,1))]
    if cropped_y.shape[-1] == 0:
        return x, y
    else:
        return cropped_x, cropped_y

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

def rotate_torch_image(image,angle):
    #use a rot90 funciton depending on type of image
    #image is a torch tensor
    if angle == 90:
        return np.rot90(image,3)
    elif angle == 180:
        return np.rot90(image,2)
    elif angle == 270:
        return np.rot90(image,1)
    return np.rot90(image,1)

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


class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, activation_fn, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = activation_fn(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, activation_fn, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = activation_fn(inputs)    
        #inputs = torch.tensor(torch.where(inputs>0.5,1.0,0.0),requires_grad=True)   
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice
    
class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, activation_fn, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = activation_fn(inputs)    
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return 1 - IoU

class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, activation_fn, smooth=1):
        
        alpha = 0.8
        gamma = 2

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = activation_fn(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #first compute binary cross-entropy 
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
                       
        return focal_loss

class TverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(TverskyLoss, self).__init__()

    def forward(self, inputs, targets, activation_fn, smooth=1, alpha=0.5, beta=0.5):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = activation_fn(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
       
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        
        return 1 - Tversky

class FocalTverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=0.5, beta=0.5, gamma=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
        
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        FocalTversky = (1 - Tversky)**gamma
                       
        return FocalTversky
    
class MyDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform
        #self.device = "cuda:0"
        
    def __getitem__(self, index):
        
        x = self.data[index]
        y = self.targets[index]
        
        #To tensor
        x = torch.from_numpy(x.copy())
        y = torch.from_numpy(y.copy())

        #To device
        x = x.to("cuda:0")
        y = y.to("cuda:0")      
        
        return x, y
    
    def __len__(self):
        return len(self.data)
    
def train_sweep(config=None):
    with wandb.init(config=config):
        config = wandb.config

        model = UNet()
        model = model.to('cuda:0')

        if config.dataset == 'cellpose generalist':
            X_train = X_train_cg
            X_test = X_test_cg
            y_train = y_train_cg
            y_test = y_test_cg
        elif config.dataset == 'cellpose trained':
            X_train = X_train_ct
            X_test = X_test_ct
            y_train = y_train_ct
            y_test = y_test_ct
        elif config.dataset == 'cellprob model':
            X_train = X_train_cm
            X_test = X_test_cm
            y_train = y_train_cm
            y_test = y_test_cm

        if config.loss_function == 'DiceBCELoss':
            loss_fn = DiceBCELoss()
        elif config.loss_function == 'BCELoss':
            loss_fn = nn.BCELoss()
        elif config.loss_function == 'DiceLoss':
            loss_fn = DiceLoss()
        elif config.loss_function == 'IoULoss':
            loss_fn = IoULoss()
        elif config.loss_function == 'FocalLoss':
            loss_fn = FocalLoss()
        elif config.loss_function == 'TverskyLoss':
            loss_fn = TverskyLoss()
        elif config.loss_function == 'FocalTverskyLoss':
            loss_fn = FocalTverskyLoss()

        if config.optimiser == 'Adam':
            optimiser = Adam(model.parameters(), lr=config.learning_rate)
        elif config.optimiser == 'SGD':
            optimiser = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
        elif config.optimiser == 'RMSprop':
            optimiser = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate)

        if config.activation_function == 'sigmoid':
            activation_fn = torch.sigmoid
        elif config.activation_function == 'relu':
            activation_fn = torch.relu
        elif config.activation_function == 'tanh':
            activation_fn = torch.tanh

        train_loader = torch.utils.data.DataLoader(list(zip(X_train,y_train)), batch_size=config.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(list(zip(X_test,y_test)), batch_size=config.batch_size, shuffle=True)

        for epoch in range(config.epochs):
            dice = train_epoch(model, train_loader, test_loader, loss_fn, activation_fn, optimiser)
            wandb.log({'dice coefficient': dice,"epoch": epoch})


def train_epoch(model, train_loader, test_loader, loss_fn, activation_fn, optimiser):
    model.train()

    #get train loss
    total_train_loss_per_epoch = 0
    for i, (x, y) in enumerate(train_loader):


        x = x.type(torch.float32)
        y = y.type(torch.float32)
        (x,y) = (x.to('cuda:0'), y.to('cuda:0')) # sending the data to the device (cpu or GPU)
        x = x.unsqueeze(1)
        pred = model(x)# make a prediction
        loss = loss_fn(pred, y, activation_fn) # calculate the loss of that prediction
        optimiser.zero_grad() # zero out the accumulated gradients
        loss.backward() # backpropagate the loss
        optimiser.step() # update model parameters
        
        total_train_loss_per_epoch += loss.detach().item()
    total_train_loss_per_epoch /= len(train_loader)
   
    #get test loss
    total_test_loss_per_epoch = 0
    total_dice = 0
    with torch.no_grad():
        for images, cellprobs in test_loader:
            images = torch.unsqueeze(images,1)
            cellprobs = torch.unsqueeze(cellprobs,1)
            cellprobs = cellprobs.to(torch.float32)
            outputs = model(images)

            #outputs = activation_fn(outputs)
            loss = loss_fn(outputs, cellprobs, activation_fn)
            total_test_loss_per_epoch += loss.item()

            #calculate dice score
            outputs = activation_fn(outputs)
            outputs = torch.where(outputs>0.5,1.0,0.0)
            outputs = outputs.view(-1)
            cellprobs = cellprobs.view(-1)
            intersection = (outputs * cellprobs).sum()  
            dice = (2.*intersection+1)/(outputs.sum() + cellprobs.sum()+1)  
            total_dice += dice.item()
            

    total_test_loss_per_epoch /= len(test_loader)
    total_dice /= len(test_loader)

    return total_dice

sweep_config = {
    'method': 'grid',
    'metric': {
        'name': 'dice coefficient',
        'goal': 'maximize'
    },
    'parameters': {
        'epochs':{
            'value': 100
        },
        'learning_rate': {
            'values': [0.1,0.01,0.001,0.0001,0.00001,0.000001]
        },
        'batch_size': {
            'values': [1,2,4,8,16,32,64,128,256,512,1024]
        },
        'activation_function': {
            'values': ['sigmoid','=relu','tanh']
        },
        'optimiser': {
            'values': ['Adam','SGD','RMSprop']
        },
        'loss_function': {
            'values': ['BCELoss','DiceLoss','DiceBCELoss','IoULoss','FocalLoss','TverskyLoss','FocalTverskyLoss']
        },
        'dataset': {
            'values': ['cellpose generalist','cellpose trained','cellprob model']
        }
    }
}

if __name__ == '__main__':
    images_path = 'data/'
    images = import_images_from_path(images_path,num_imgs=23,normalisation=True)

    #get groundtruth masks from data_for_cellpose
    #import the numpy files from the masks folder and binarise them
    masks = []
    for i in range(23):
        mask = np.load(os.getcwd()+'\\masks'+'\\'+str(i)+'_seg.npy',allow_pickle=True).item()['masks']
        mask = np.where(mask>0.5,1,0) #binarise the masks
        masks.append(mask)

    #Getting input data from Cellpose trained
    model_path = 'models/CP_20230402_212503_3'
    model = import_cellpose_model(model_path)
    cell_probabilities_cellpose_trained = get_cellpose_probability_maps_pre_trained(model,images)
    cell_probabilities_cellpose_trained = [(cellprob-np.min(cellprob))/(np.max(cellprob)-np.min(cellprob)) for cellprob in cell_probabilities_cellpose_trained]
    cellprob_crops_ct, mask_crops_ct = get_random_crops_from_multiple_images(cell_probabilities_cellpose_trained,masks,num_crops=1000)
    X_train_ct, X_test_ct, y_train_ct, y_test_ct = train_test_split(cellprob_crops_ct, mask_crops_ct, test_size=0.33, random_state=42)
    X_train_ct, y_train_ct = rotate_images_and_cellprobs_return_merged(X_train_ct,y_train_ct,angles=[90,180,270])

    #Getting input data from Cellpose generalist
    cell_probabilities_cellpose_generalist = get_cellpose_probability_maps(images)
    cell_probabilities_cellpose_generalist = [(cellprob-np.min(cellprob))/(np.max(cellprob)-np.min(cellprob)) for cellprob in cell_probabilities_cellpose_generalist]


    