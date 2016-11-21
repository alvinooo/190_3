import cPickle, gzip
import numpy as np
import math
#import matplotlib.pyplot as plt
from keras.layers import Dropout
from keras.models import load_model
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Reshape, Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.callbacks import Callback
#import cropping
from keras.layers.normalization import BatchNormalization

size = 50000
batch_size = 32
n_epoch = 10 
drop_percent = 0.25
n_categories = 10

def load_data():

    datasetCifar = cifar10.load_data()
    train_data = datasetCifar[0][0][:size]
    train_labels = datasetCifar[0][1][:size]
    test_data = datasetCifar[1][0][:size]
    test_labels = datasetCifar[1][1][:size]

    return train_data, train_labels, test_data, test_labels


class TestLoss(Callback):
    def __init__(self, test_data, test_plot, model,datagen):
        self.test_data = test_data
        self.model = model
        self.test_plot = test_plot
	self.datagen = datagen

    def on_epoch_end(self, batch, logs={}):
        X_test, T_test_onehot = self.test_data
        loss, acc = self.model.evaluate_generator(self.datagen.flow(X_test,T_test_onehot, batch_size=32),val_samples=len(X_test) )
        #self.test_plot.append((loss, acc))
        print "loss: ", loss
        print "acc: ", acc



def preprocess_data(X_train):
   datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    vertical_flip=True) 

   datagen.fit(X_train)
   return datagen   

def preprocess_test_data(X_train):
    datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True)

    datagen.fit(X_train)
    return datagen

def build_net():
    (X,T,X_test,T_test) = load_data()

    T_onehot = to_categorical(T,n_categories)
    T_test_onehot = to_categorical(T_test, n_categories)
    print "X", X[0]
    X = X.astype(float)
    print "X", X[0]
    X_test = X_test.astype(float)
    datagen = preprocess_data(X)
    datagen_test = preprocess_test_data(X_test)
    
    model = load_model("fine_tune.h5")

    model.pop()
    model.summary()

    for i in model.layers[:9]:
        i.trainable = False



    #Output Softmax Layer
    model.add(Dense(10, init='normal', activation="softmax", name="dense_new"))
    model.summary()
 
    #Train
    model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    test_plot = []

    train_plot = model.fit_generator(datagen.flow(X, T_onehot, batch_size=32),
                    samples_per_epoch=len(X), nb_epoch=n_epoch, callbacks=[TestLoss((X_test, T_test_onehot), test_plot, model, datagen_test)])

   
    


build_net()

