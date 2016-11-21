import cPickle, gzip
import numpy as np
import math
#import matplotlib.pyplot as plt
from keras.layers.core import Dropout
from keras.datasets import cifar100
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D
#from keras.layers.
from keras.layers.pooling import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.callbacks import Callback
#import cropping


size = 10000

def load_data():

    datasetCifar = cifar100.load_data()
    train_data = datasetCifar[0][0][:size]
    #train_data = train_data.transpose((0, 3, 1, 2))
    train_labels = datasetCifar[0][1][:size]
    test_data = datasetCifar[1][0][:size]
    test_labels = datasetCifar[1][1][:size]
    
    return train_data, train_labels, test_data, test_labels


class TestLoss(Callback):
    def __init__(self, test_data, test_plot, model, datagen):
        self.test_data = test_data
	self.model = model
	self.test_plot = test_plot
	self.datagen = datagen

    def on_epoch_end(self, batch, logs={}):
        X_test, T_test_onehot = self.test_data
        loss, acc = self.model.evaluate(X_test,T_test_onehot, batch_size=32)
        #self.test_plot.append((loss, acc))
    	print "loss: ", loss
	print "acc: ", acc
def build_net():
    (X,T,X_test,T_test) = load_data()

    T_onehot = to_categorical(T,100)
    T_test_onehot = to_categorical(T_test, 100)

    model = Sequential()
    # model.add(BatchNormalization(input_shape=(32, 32, 3)))
    print "Conv Layer 1"
    #ConvLayer 1
    model.add(Convolution2D(32, 3, 3, init='normal', border_mode='same', input_shape=(32, 32, 3)) )
    model.add(Dropout(0.2))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.summary()
 
    print "Conv Layer 2"
    #ConvLayer 2
    model.add(Convolution2D(64, 3, 3, init='normal', border_mode='same'))
    model.add(Dropout(0.3))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.summary()
    
    print "Conv Layer 3"
    #ConvLayer 3
    model.add(Convolution2D(64, 3, 3, init='normal', border_mode='same'))
    model.add(Dropout(0.3))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.summary()
    
    #Fully connected layer
    model.add(Flatten())
    model.add(Dense(100, init='normal'))
    model.add(Dropout(0.2))
    model.add(Activation("relu"))
    model.summary()

    #Output Softmax Layer
    model.add(Dense(100, init='normal', activation="softmax"))
    model.summary()
    
    #print model.layers[0].input_shape

    
    #Train
    model.compile(optimizer='adadelta',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    
    test_plot = []
    train_plot = model.fit(X,T_onehot,nb_epoch=100, callbacks=[TestLoss((X_test,T_test_onehot), test_plot, model)],batch_size=32)    

    print "model.fit"
    print "tet_plot", test_plot
    

build_net()
