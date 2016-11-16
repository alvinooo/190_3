import cPickle, gzip
import numpy as np
import math
#import matplotlib.pyplot as plt
from keras.datasets import cifar100
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils.np_utils import to_categorical
from keras.callbacks import Callback
#import cropping

def load_data():

    datasetCifar = cifar100.load_data()
    train_data = datasetCifar[0][0]
    train_labels = datasetCifar[0][1]
    test_data = datasetCifar[1][0]
    test_labels = datasetCifar[1][1]
    
    return train_data, train_labels, test_data, test_labels

class TestLoss(Callback):
    def __init__(self, test_data, test_plot,model):
        super(Callback,self).__init__()
        self.test_data = test_data
        self.model = model
        self.test_plot = test_plot

    def on_epoch_end(self, batch, logs={}):
        X_test, T_test_onehot = self.test_data
        loss, acc = self.model.evaluate(X_test,T_test_onehot, batch_size=32)
        self.test_plot.append((loss, acc))
"""
def show_image(ind):
    image = train_data[ind,:,:,:]
    plt.imshow(image)
    plt.show()
""" 
def build_net():
    (X,T,X_test,T_test) = load_data()

    #X = X[0 : 1000,:]
    #T = T[0 : 1000, :]

    T_onehot = to_categorical(T,100)
    T_test_onehot = to_categorical(T_test,100)

    model = Sequential()
    # model.add(BatchNormalization(input_shape=(32, 32, 3)))

    #ConvLayer 1
    model.add(Convolution2D(32, 3, 3, init='normal', input_shape=(32, 32,3)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    #ConvLayer 2
    model.add(Convolution2D(64, 3, 3, init='normal', border_mode="same"))
   # model.add(BatchNormalization())
   # model.add(Activation("relu"))
   # model.add(MaxPooling2D(pool_size=(2, 2)))
    
    #ConvLayer 3
   # model.add(Convolution2D(128, 3, 3, init='normal'))
   # model.add(BatchNormalization())
   # model.add(Activation("relu"))
   # model.add(MaxPooling2D(pool_size=(2, 2)))
    
    #Fully connected layer
    model.add(Flatten())
    model.add(Dense(15*15*64, init='normal'))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    #Output Softmax Layer
    model.add(Dense(100, init='normal', activation="softmax"))
    
    model.summary()
    #Train
    model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    
    test_plot = []
    train_plot = model.fit(X,T_onehot,nb_epoch=100, callbacks=[TestLoss((X_test,T_test_onehot), test_plot, model)],batch_size=32)
"""    
    plt.plot(train_plot.history['acc'])

    plt.plot(test_plot)
   
    #model.metrics_names

    #calculate mean
    mean = np.mean(train_data)
    
    #go through all pixels and sub mean
    train_data = (train_data - mean)/mean
"""    
    # maybe not -> https://piazza.com/class/iteh1o6zzoa2wy?cid=345    
build_net()    

