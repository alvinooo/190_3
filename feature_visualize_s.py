from scipy.misc import imsave
from keras import backend as K
from keras.models import load_model
import cPickle, gzip
import numpy as np
import math
#import matplotlib.pyplot as plt
from keras.layers import Dropout
from keras.datasets import cifar100
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.callbacks import Callback
#import cropping
from keras.layers.normalization import BatchNormalization

# util function to convert a tensor into a valid image
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def build_net():
    
    model = load_model("cifar100_best.h5")
    model.summary()
    layer_dict = dict([(layer.name, layer) for layer in model.layers])

    layer_name = "convolution2d_1" 
    filter_index = 0 
    input_img = model.layers[0].input

    layer_output = layer_dict[layer_name].output
    loss = K.mean(layer_output[:, filter_index, :, :])
    grads = K.gradients(loss, input_img)[0]
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    iterate = K.function([input_img], [loss, grads])
    
    input_img_data = np.random.random((1, 32, 32, 3)) * 20 + 128 
    for i in range(20):
     loss_value, grads_value = iterate([input_img_data])
     input_img_data += grads_value * 0.01
    
    img = input_img_data[0]
    img = deprocess_image(img)
    imsave('%s_filter_%d.png' % (layer_name, filter_index), img)


    layer_name = "convolution2d_2"
    


build_net()

