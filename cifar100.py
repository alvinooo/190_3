import cPickle, gzip
import numpy as np
import math
import csv
from summary import summary
from keras.datasets import cifar100
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, SpatialDropout2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

categories = 100
data_size = 50000
data = ImageDataGenerator(featurewise_center=True,
                          featurewise_std_normalization=True,
                          width_shift_range=0.1,
                          height_shift_range=0.1,
                          horizontal_flip=True)
test = ImageDataGenerator(featurewise_center=True,
                          featurewise_std_normalization=True)

def load_data():

    datasetCifar = cifar100.load_data()
    x = datasetCifar[0][0][:data_size]
    y = datasetCifar[0][1][:data_size]
    tx = datasetCifar[1][0][:data_size]
    ty = datasetCifar[1][1][:data_size]
    return x, y, tx, ty

def show_image(ind):
    image = train_data[ind,:,:,:]
    # plt.imshow(image)
    # plt.show()

def subset(x, y, size):
    category_counts = {}
    sub_x, sub_y = [], []
    for (image, label_array) in zip(x, y):
        label = label_array[0]
        if label not in category_counts:
            category_counts[label] = 0
        if category_counts[label] < size:
            category_counts[label] += 1
            sub_x.append(image)
            sub_y.append(label_array)
    return np.array(sub_x), np.array(sub_y)

def add_convolution(model, feature_maps, input_shape=(), pool=True):
    model.add(Convolution2D(feature_maps, 3, 3,
                            init='normal',
                            border_mode='same',
                            input_shape=input_shape))
    model.add(BatchNormalization(mode=0, axis=1))
    model.add(Activation("relu"))
    if pool:
        model.add(MaxPooling2D(pool_size=(2, 2)))

def add_fully(model, hidden_units, dropout):
    model.add(Flatten())
    model.add(Dense(hidden_units, 'glorot_normal'))
    model.add(BatchNormalization(mode=0, axis=1))
    model.add(Activation("relu"))
    model.add(Dropout(dropout))

def build_net(dropout=0):
    model = Sequential()
    add_convolution(model, 32, input_shape=(32, 32, 3))
    add_convolution(model, 64)
    add_convolution(model, 128)
    add_fully(model, 512, dropout)
    model.add(Dense(categories, 'glorot_normal', activation="softmax"))
    model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    return model

def build_shallow_net(dropout=0):
    model = Sequential()
    add_convolution(model, 8, input_shape=(32, 32, 3))
    add_fully(model, 512, dropout)
    model.add(Dense(categories, 'glorot_normal', activation="softmax"))
    model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    return model

def build_deep_net(dropout=0):
    model = Sequential()

    add_convolution(model, 64, input_shape=(32, 32, 3))
    add_convolution(model, 128)
    add_convolution(model, 128)
    add_convolution(model, 256)
    add_fully(model, 512, dropout)
    model.add(Dense(categories, 'glorot_normal', activation="softmax"))
    model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    return model

def train_net(model, epochs, datagen, testgen, size=None, h5_path=None):
    (x, y, tx, ty) = load_data()
    if size:
        x, y = subset(x, y, size)
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
    if h5_path:
        model.save(h5_path)
    return plot

def compare(epochs):

    output = []

    # Default
    # model_default = build_net()
    # plot_default = train_net(model_default, epochs,
    #                          ImageDataGenerator(), ImageDataGenerator())
    # output.append(plot_default.history['loss'])
    # output.append(plot_default.history['acc'])
    # output.append(plot_default.history['val_loss'])
    # output.append(plot_default.history['val_acc'])

    # # Optimized
    # model_best = build_net(dropout=0.5)
    # plot_best = train_net(model_best, epochs, data, test,
    #                       h5_path="cifar100_best.h5")
    # output.append(plot_best.history['loss'])
    # output.append(plot_best.history['acc'])
    # output.append(plot_best.history['val_loss'])
    # output.append(plot_best.history['val_acc'])

    # # 100 per category
    # model_100 = build_net(dropout=0.5)
    # plot_100 = train_net(model_100, epochs, data, test, size=100)
    # output.append(plot_100.history['loss'])
    # output.append(plot_100.history['acc'])
    # output.append(plot_100.history['val_loss'])
    # output.append(plot_100.history['val_acc'])
    #
    # # 300 per category
    # model_300 = build_net(dropout=0.5)
    # plot_300 = train_net(model_300, epochs, data, test, size=300)
    # output.append(plot_300.history['loss'])
    # output.append(plot_300.history['acc'])
    # output.append(plot_300.history['val_loss'])
    # output.append(plot_300.history['val_acc'])
    #
    # # Shallow
    # model_shallow = build_shallow_net(dropout=0.5)
    # plot_shallow = train_net(model_shallow, epochs, data, test)
    # output.append(plot_shallow.history['loss'])
    # output.append(plot_shallow.history['acc'])
    # output.append(plot_shallow.history['val_loss'])
    # output.append(plot_shallow.history['val_acc'])

    # # Deep
    # model_deep = build_deep_net(dropout=0.5)
    # plot_deep = train_net(model_deep, epochs, data, test,
    #                       h5_path="cifar100_deep.h5")
    # output.append(plot_deep.history['loss'])
    # output.append(plot_deep.history['acc'])
    # output.append(plot_deep.history['val_loss'])
    # output.append(plot_deep.history['val_acc'])

    # with open('output.csv', 'w') as f:
    #     writer = csv.writer(f)
    #     for l in output:
    #         writer.writerow(l)

model_best = build_net(dropout=0.5)
summary(model_best)
compare(50)