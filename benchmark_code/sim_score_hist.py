# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 17:32:01 2019

Test on the CampusLoopDataset, plot the similarity score
histogam plot of the correct match.
"""


import numpy as np
import cv2
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from keras import models
from AcrossChannelNorm import LRN


def resizeImg(img, rw=160, rh=120):
    img = cv2.resize(img, (rw, rh), interpolation = cv2.INTER_CUBIC)   # resize the image
    return img


def predict_on_keyframe(keyframe, model, preprocess):
    '''
    For a given keyframe (RGB image), use the pretrained model to
    predict the HOG feature of the keyframe.
    Params:
        keyframe: a RGB image in numpy array format
        model: pretrained keras model object
    Output:
        desciptor: a 1D numpy array 1*936
    '''
    
    if preprocess==True:
        img_yuv = cv2.cvtColor(keyframe, cv2.COLOR_RGB2YUV)
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        keyframe = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    
    # conver keyframe into gray scale
    gray = cv2.cvtColor(keyframe, cv2.COLOR_RGB2GRAY)
    # resize to fot into network input size 120*160
    gray = resizeImg(gray)
   
    gray = gray / 255.0
    # expand gray format from h*w to h*w*1
    x = np.expand_dims(gray, axis=-1)
    
    list = []
    list.append(x)
    # expand from h*w*1 to 1*h*w*1
    x = np.stack(list, axis=0)
    
    # predict the hog feature
    descriptor = model.predict(x) 
    
    return descriptor


def predict_on_CampusLoopDataset():
    # path variables
    base = 'E:/UnderRoot/my_dataset/CampusLoopDataset/'
    imgs1_path = base + 'live'
    out1_path = base + 'livehogs'
    imgs2_path = base + 'memory'
    out2_path = base+ 'memoryhogs'
    model_path = 'E:/Github/na568-project-team2-master/models/calc_model_6Million.h5'
    preprocess = True
    
    # load the pretrained network model
    print('Loading the trained model from: ', model_path)
    my_model = models.load_model(model_path,custom_objects={'LRN': LRN})
    print('Only grap the descriptor layer.')
    deploy_model = models.Model(inputs=my_model.input,
            outputs=my_model.get_layer('deploy').output)
    
    # predict on folder CampusLoopDataset/live
    names = os.listdir(imgs1_path)
    num = len(names)
    for i in range(0, num):
        keyframe = mpimg.imread(imgs1_path+'/'+names[i])
        descriptor = predict_on_keyframe(keyframe, deploy_model, preprocess)
        descriptor = descriptor / np.linalg.norm(descriptor)
        np.save(out1_path+'/'+str(i+1)+'.npy',descriptor)
        if i%4 == 0:
            print('Save prediction results for %s'%('live/'+names[i]))
        
    # predict on folder CampusLoopDataset/live
    names = os.listdir(imgs2_path)
    num = len(names)
    for i in range(0, num):
        keyframe = mpimg.imread(imgs2_path+'/'+names[i])
        descriptor = predict_on_keyframe(keyframe, deploy_model, preprocess)
        descriptor = descriptor / np.linalg.norm(descriptor)
        np.save(out2_path+'/'+str(i+1)+'.npy',descriptor)
        if i%4 == 0:
            print('Save prediction results for %s'%('memory/'+names[i]))
    
    return


def load_DescrSet(path):
    names = os.listdir(path)
    num = len(names)
    
    descrs = []
    for i in range(0, num):
        descr= np.load(path+'/'+names[i])
        descrs.append(descr)
    
    descrset = np.concatenate(descrs,axis=0)
    
    return descrset
    

def l2_distance(x1, x2):
    '''
    compute the euclidean distance between to 1D numpy array
    '''
    dist = np.sum(np.square(x1-x2), axis=0)
    dist = np.sqrt(dist)
    return dist


def compute_sim_score(x, Y):
    n = Y.shape[0]   # number of row vectors in Y
    scores = np.zeros(n)
    for i in range(0, n):
        scores[i] = np.dot(Y[i,:], x.T)
     
    # normalize the score to let the minimum is 0 and
    # maximum is 1.0
    norm_scores = cv2.normalize(scores, None, 0.0, 1.0, cv2.NORM_MINMAX)
    
    return norm_scores




if __name__ == '__main__':
    
    predict_on_CampusLoopDataset()
    
    livedescr_path = 'E:/UnderRoot/my_dataset/CampusLoopDataset/livehogs'
    memorydescr_path = 'E:/UnderRoot/my_dataset/CampusLoopDataset/memoryhogs'
    
    set1 = load_DescrSet(livedescr_path)
    set2 = load_DescrSet(memorydescr_path)
    
    img_idx = 42
    scores = compute_sim_score(set1[img_idx-1,:],set2)
    print(scores.shape)
    x = np.arange(1,101)

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)
    ax.bar(x, scores[:,0], align='center',width=0.6)
