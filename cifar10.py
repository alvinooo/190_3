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
#import cropping

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
    model = load_model("cifar100_best.h5")
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
print plot.history['acc']
print plot.history['val_acc']

# plt.plot(train_plot.history['acc'])
# plt.plot(test_plot)
#model.metrics_names

# 1e-3
# [0.45145999999999997, 0.59384000000000003, 0.62048000000000003, 0.63482000000000005, 0.64856000000000003, 0.65207999999999999, 0.65925999999999996, 0.66279999999999994, 0.66944000000000004, 0.67312000000000005, 0.67789999999999995, 0.68001999999999996, 0.68218000000000001, 0.68430000000000002, 0.68235999999999997, 0.68723999999999996, 0.68940000000000001, 0.68872, 0.68725999999999998, 0.69457999999999998, 0.69501999999999997, 0.69218000000000002, 0.69999999999999996, 0.69674000000000003, 0.70001999999999998, 0.70018000000000002, 0.70374000000000003, 0.70257999999999998, 0.70050000000000001, 0.70199999999999996, 0.70250000000000001, 0.70187999999999995, 0.70272000000000001, 0.70433999999999997, 0.70598000000000005, 0.70867999999999998, 0.70816000000000001, 0.70713999999999999, 0.70835999999999999, 0.71052000000000004, 0.71167999999999998, 0.71294000000000002, 0.71104000000000001, 0.71492, 0.71748000000000001, 0.71284000000000003, 0.71526000000000001, 0.71350000000000002, 0.71430000000000005, 0.71772000000000002]
# [0.64590000000000003, 0.68330000000000002, 0.69920000000000004, 0.7147, 0.72089999999999999, 0.72550000000000003, 0.72929999999999995, 0.73370000000000002, 0.73799999999999999, 0.74299999999999999, 0.74629999999999996, 0.74129999999999996, 0.75519999999999998, 0.74490000000000001, 0.75080000000000002, 0.755, 0.75339999999999996, 0.75639999999999996, 0.75249999999999995, 0.75870000000000004, 0.75970000000000004, 0.75819999999999999, 0.76380000000000003, 0.75929999999999997, 0.75890000000000002, 0.76259999999999994, 0.76919999999999999, 0.76100000000000001, 0.76129999999999998, 0.76529999999999998, 0.76770000000000005, 0.76259999999999994, 0.76590000000000003, 0.76619999999999999, 0.77100000000000002, 0.76339999999999997, 0.76570000000000005, 0.7681, 0.76590000000000003, 0.7772, 0.76600000000000001, 0.77190000000000003, 0.76639999999999997, 0.77080000000000004, 0.77270000000000005, 0.77410000000000001, 0.76970000000000005, 0.77180000000000004, 0.77529999999999999, 0.77200000000000002]