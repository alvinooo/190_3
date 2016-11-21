# Imports
#import matplotlib.pyplot as plt

import theano

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Activation


# Model 
model = Sequential()

model.add(Convolution2D(28, 1, 3, 3, border_mode='same')) 
convout1 = Activation('relu')
model.add(convout1)


# Data loading + reshape to 4D
(X_train, y_train), (X_test, y_test) = mnist_dataset = mnist.load_data()
reshaped = X_train.reshape(X_train.shape[0], 1, X_train.shape[1], X_train.shape[2])


from random import randint
img_to_visualize = randint(0, len(X_train) - 1)


# Generate function to visualize first layer
convout1_f = theano.function([model.get_input(train=False)], convout1.get_output(train=False))
convolutions = convout1_f(reshaped[img_to_visualize: img_to_visualize+1])

