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

num_hidden = 1
categories = 10

def load_data():

    datasetCifar = cifar100.load_data()
    train_data = datasetCifar[0][0][:10]
    train_labels = datasetCifar[0][1][:10] / 100
    test_data = datasetCifar[1][0][:10]
    test_labels = datasetCifar[1][1][:10] / 100
    # print train_data.shape
    # print train_labels.shape
    # print test_data.shape
    # print test_labels.shape

    return train_data, train_labels, test_data, test_labels

class TestLoss(Callback):
    def __init__(self, model, test_data, test_plot):
        self.model = model
        self.test_data = test_data
        self.test_plot = test_plot

    def on_epoch_end(self, batch, logs={}):
        X_test, T_test_onehot = self.test_data
        loss, acc = self.model.evaluate(X_test,T_test_onehot, batch_size=32)
        self.test_plot.append((loss, acc))

def show_image(ind):
    image = train_data[ind,:,:,:]
    # plt.imshow(image)
    # plt.show()

def build_net():
    (X,T,X_test,T_test) = load_data()

    T_onehot = to_categorical(T, nb_classes=categories)
    T_test_onehot = to_categorical(T_test, nb_classes=categories)

    model = Sequential()
    # model.add(BatchNormalization(input_shape=(32, 32, 3)))

    #ConvLayer 1
    model.add(Convolution2D(32, 3, 3, init='normal', border_mode='same', input_shape=(32, 32, 3)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #ConvLayer 2
    model.add(Convolution2D(64, 3, 3, init='normal', border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # #ConvLayer 3
    # model.add(Convolution2D(128, 3, 3, init='normal', border_mode='same'))
    # model.add(BatchNormalization())
    # model.add(Activation("relu"))
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    #Fully connected layer
    model.add(Flatten())
    model.add(Dense(num_hidden, 'glorot_normal'))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    #Output Softmax Layer
    model.add(Dense(categories, 'glorot_normal', activation="softmax"))

    #Train
    model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    # print type(model.layers[-4])
    # print model.layers[-4].output_dim
    # print model.layers[-1].output_dim
    test_plot = []
    # print X.shape
    # print model.layers[0].input_shape
    # print model.layers[5].input_shape
    print model.layers[0].input_shape
    print model.layers[4].input_shape
    # print model.layers[8].input_shape
    # print model.layers[13].input_shape
    # print model.layers[16].input_shape
    train_plot = model.fit(X,T_onehot,nb_epoch=1,
                           callbacks=[TestLoss(model, (X_test,T_test_onehot),
                                               test_plot)],batch_size=32)
    # plt.plot(train_plot.history['acc'])

    # plt.plot(test_plot)

    #model.metrics_names

    # #calculate mean
    # mean = np.mean(train_data)
    #
    # #go through all pixels and sub mean
    # train_data = (train_data - mean)/mean

    # maybe not -> https://piazza.com/class/iteh1o6zzoa2wy?cid=345

build_net()