# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 23:37:40 2019

@author: Kun Sun
"""


import numpy as np
import random
from keras import models, layers, optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from AcrossChannelNorm import LRN


def genBatch_fromFile(data_dir, batch_size, num, usage, rescale):
    '''
    Flow a batch of images (with labels) from directory for 
    using keras.models.fit_generator().
    Params:
        data_dir: path containing folders "train/val" and "labels"
        batch_size: number of images to generate in a batch 
        num: total number of images in the training/validation set
        usage: either 'train' or 'val'
        rescale: whether to rescale the images
    Output:
        a tuple (images, labels)
    '''
    
    while True:
        idx = random.sample(range(1,num+1),batch_size)   # randomly pick a batch of image indexes
        img_list = []
        label_list = []
        for i in idx:
            img = np.load(data_dir+'/'+usage+'/'+str(i)+'.npy')
            label = np.load(data_dir+'/'+usage+'labels/'+str(i)+'.npy')
            img_list.append(img)
            label_list.append(label)
        batch_imgs = np.stack(img_list, axis=0)
        batch_labels = np.stack(label_list, axis=0) 
        
        if rescale == True:
            batch_imgs = batch_imgs * 1.0 / 255.0
        
        yield (batch_imgs, batch_labels)
        

def buildNet(batch_size, w=160, h=120, c=1):
    '''
    Build a CNN model object which converts the input image into a HOG vector. 
    '''
    
    input_imgs = layers.Input(shape=(h, w, c))
    
    # downsample(encode)
    x = layers.ZeroPadding2D(padding=(4,4))(input_imgs)
    conv1 = layers.Conv2D(64,(5,5),activation='relu',strides=(2,2),padding ='valid')(x)
    pool1 = layers.MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='valid')(conv1)
    norm1 = LRN(0.0001,0.75,1,5)(pool1)
    
    x = layers.ZeroPadding2D(padding=(2,2))(norm1)
    conv2 = layers.Conv2D(128,(4,4),activation='relu',strides=(1,1),padding ='valid')(x)
    pool2 = layers.MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='valid')(conv2)
    norm2 = LRN(0.0001,0.75,1,5)(pool2)
    
    conv3 = layers.Conv2D(4,(3,3),activation='relu',strides=(1,1),padding ='valid')(norm2)
    
    descriptor = layers.Flatten()(conv3)
    
    # upsample(decode)
    fc4 = layers.Dense(1064, activation='sigmoid')(descriptor)
    
    fc5 = layers.Dense(2048, activation='sigmoid')(fc4)
    
    predictions = layers.Dense(3648, activation='sigmoid')(fc5)   # X2_hat
    
    # set the inputs and outputs and compile the model
    our_model = models.Model(inputs=input_imgs, outputs=predictions)
    our_optimizer = optimizers.SGD(
            lr=0.0009,momentum = 0.9,decay = 0.0005,nesterov = True)
    our_model.compile(
            loss='mean_squared_error',
            optimizer = our_optimizer, metrics = ['mse'])
    
    return our_model
    

def train():
    # your folder path which containing all four folders "train", "trainlabels", "val", "vallabels"
    data_dir = ...
    # Set it to Ture if you want to continue to train a pretrained model
    start_from_checkpoint = False
    # your path to store the trained network model file 
    checkpoint_path = 'E:/JuWorkDir/568_project/calc_model.h5'
   
    ########## model parameters ##########
    # number of training images and validation images
    num_train, num_val = 25550, 10950
    input_height, input_width, channels = 120, 160, 1
    our_batch_size = 2
    our_epochs = 5
    num_train_batches = num_train // our_batch_size
    num_val_batches = num_val // our_batch_size
    
    ########## model parameters ##########
    if start_from_checkpoint == True:
       our_model = models.load_model(checkpoint_path)
       print('Start from checkpoint: ',checkpoint_path)
    else:
       our_model = buildNet(our_batch_size,input_width,input_height,channels)
       print('Start from the beginning')
    
    print(our_model.summary())
    
    ########## data pipeline ##########
    trainGen = genBatch_fromFile(data_dir,our_batch_size,num_train,'train',True)
    valGen = genBatch_fromFile(data_dir,our_batch_size,num_val,'val',True)
    
    ########## checkpoint and model save ##########
    checkpoint = ModelCheckpoint(
            checkpoint_path,monitor='val_loss',verbose=1,
            save_best_only=True,save_weights_only=False,mode='auto',period=1)
    early_stop = EarlyStopping(monitor='val_loss',
                            min_delta=0,patience=5,verbose=1,mode='auto')
    
    ########## model training ##########
    our_model.fit_generator(
        trainGen,   # the generator outputs a tuple of (inputs, targets)
        steps_per_epoch = num_train_batches,
        epochs = our_epochs,
        validation_data = valGen,   # the generator outputs a tuple of (x_val, y_val)
        validation_steps = num_val_batches,
        callbacks = [checkpoint, early_stop],
        initial_epoch=0)
    
    return
    

if __name__ == '__main__':
    train()
    
