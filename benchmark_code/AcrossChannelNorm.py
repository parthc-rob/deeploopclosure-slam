# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 19:22:40 2019

@author: code is rewritten from 
https://gist.github.com/joelouismarino/b78257a716df15fb1886442927cc6d72#file-googlenet_lrn_def-py 
and https://github.com/BVLC/caffe/blob/master/src/caffe/layers/lrn_layer.cpp

This file defines the old-fashioned Local Region Response(LRN) layer which has
been discarded by both Keras and Pytorch. Only ACROSS_CHANNEL model is supported
here. 
"""


from keras import backend as K
from keras.layers import Layer
import numpy as np


class LRN(Layer):

    def __init__(self, alpha=0.0001, beta=0.75, k=1, n=5, **kwargs):
        #self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.n = n   # local region size
        super(LRN, self).__init__(**kwargs)

    def build(self, input_shape):
        super(LRN, self).build(input_shape)  

    def call(self, x):
        '''
        Divide each pixel value of x by a normalization parameter.  
        params:
            x is a 4D keras tensor with channel_last data format, i.e.,
            a batch of input images
        return:
            4D tensor x 
        '''
        
        #num_ = np.int32(self.batch_size)
        height_, width_, channels_ = x.shape[1::]
        half_n = self.n // 2
        x_sqr = K.square(x)   # perform elementwise square for input x
        
        
        '''
        padding = np.zeros((num_,height_,width_,half_n),dtype=np.float32)
        values = np.concatenate(
                [padding,
                 K.eval(x_sqr),
                 padding],
                axis=3)
        padded_sqr = K.variable(values)
        '''
        
        # sum over adjacent channels
        '''
        channel_maps = []
        for c in range(0,channels_):
            c_map = K.sum(padded_sqr[:,:,:,c:c+self.n], axis=3)
            channel_maps.append(c_map)
        scale = K.stack(channel_maps, axis=3)
        '''
        channel_maps = []
        for c in range(0,channels_):
            head = max(c-half_n, 0)
            tail = min(c+half_n, channels_)
            c_map = K.sum(x_sqr[:,:,:,head:c+tail+1], axis=3)
            channel_maps.append(c_map)
        scale = K.stack(channel_maps, axis=3)
            
        norm_alpha = self.alpha / self.n
        scale = scale * norm_alpha   # multiply the tensor by a scalar
        scale = scale + self.k   # add the tensor by a scalar
        scale = scale ** self.beta   # perform element wise expotential 
        
        x = x / scale
        return x

    def compute_output_shape(self, input_shape):
        return input_shape
    
    def get_config(self):
        config = {"alpha": self.alpha,
          "k": self.k,
          "beta": self.beta,
          "n": self.n}
        base_config = super(LRN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))