import cPickle, gzip
import numpy as np
import math
#import matplotlib.pyplot as plt
from keras import optimizers
from keras.models import load_model
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, SpatialDropout2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

categories = 10
batch = 32
data = ImageDataGenerator(featurewise_center=True,
                          featurewise_std_normalization=True,
                          width_shift_range=0.1,
                          height_shift_range=0.1,
                          horizontal_flip=True)
test = ImageDataGenerator(featurewise_center=True,
                          featurewise_std_normalization=True)

size = 50000

def load_data():

    datasetCifar = cifar10.load_data()
    train_data = datasetCifar[0][0][:size]
    train_labels = datasetCifar[0][1][:size]
    test_data = datasetCifar[1][0][:size]
    test_labels = datasetCifar[1][1][:size]
    return train_data, train_labels, test_data, test_labels

def show_image(ind):
    image = train_data[ind,:,:,:]
    # plt.imshow(image)
    # plt.show()

def build_net():
    model = load_model("cifar100_deep.h5")
    i = len(model.layers) - 1
    while type(model.layers[i]) != Convolution2D:
        i -= 1
    i -= 1
    while i >= 0:
        model.layers[i].trainable = False
        i -= 1
    model.pop()
    model.add(Dense(categories, 'glorot_normal', activation="softmax", name='dense_2'))

    model.compile(optimizer=optimizers.SGD(lr=1e-3, momentum=0.9),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    return model

def train_net(model, epochs, datagen, testgen):
    (x, y, tx, ty) = load_data()
    y = to_categorical(y, nb_classes=categories)
    ty = to_categorical(ty, nb_classes=categories)
    x = x.astype(float)
    tx = tx.astype(float)
    datagen.fit(x)
    testgen.fit(tx)
    plot = model.fit_generator(datagen.flow(x, y, batch_size=32),
                               validation_data=testgen.flow(tx, ty),
                               nb_val_samples=len(tx),
                               nb_epoch=epochs,
                               samples_per_epoch=len(x))
    return plot

model = build_net()
plot = train_net(model, 50, data, test)
print plot.history['loss']
print plot.history['acc']
print plot.history['val_loss']
print plot.history['val_acc']