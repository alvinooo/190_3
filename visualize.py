from keras import backend as K
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D

model = Sequential()
layers = {layer.name: layer for layer in model.layers}