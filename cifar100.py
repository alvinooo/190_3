import cPickle, gzip
import numpy as np
import math
#import matplotlib.pyplot as plt
from keras.datasets import cifar100
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, SpatialDropout2D
from keras.layers.convolutional import Convolution2D, ZeroPadding2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
#import cropping

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

def build_shallow_net():
    model = Sequential()

    model.add(Convolution2D(8, 3, 3, init='normal', border_mode='same', input_shape=(32, 32, 3)))
    model.add(BatchNormalization(mode=0, axis=1))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(model)
    model.add(softmax)

    model.summary()
    model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    return model

def build_deep_net():
    conv1, conv2, _, model, softmax = build_net_util()
    model = Sequential()

    model.add(conv1)
    model.add(Convolution2D(32, 3, 3, init='normal', border_mode='same'))
    model.add(BatchNormalization(mode=0, axis=1))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(conv2)
    model.add(Convolution2D(512, 3, 3, init='normal', border_mode='same'))
    model.add(BatchNormalization(mode=0, axis=1))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(model)
    model.add(softmax)
    model.summary()
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

    # # Default
    # model_default = build_net()
    # plot_default = train_net(model_default, epochs,
    #                          ImageDataGenerator(), ImageDataGenerator())
    # print plot_default.history['acc']
    # print plot_default.history['val_acc']

    # # Optimized
    # model_best = build_net(dropout=0.5)
    # plot_best = train_net(model_best, epochs, data, test,
    #                       h5_path="cifar100_best.h5")
    # print plot_best.history['acc']
    # print plot_best.history['val_acc']

    # 100 per category
    model_100 = build_net()
    plot_100 = train_net(model_100, epochs, data, test, size=100)
    print plot_100.history['acc']
    print plot_100.history['val_acc']

    # 300 per category
    model_300 = build_net()
    plot_300 = train_net(model_300, epochs, data, test, size=300)
    print plot_300.history['acc']
    print plot_300.history['val_acc']

    # # Shallow
    # model_shallow = build_shallow_net()
    #
    # # Deep
    # model_deep = build_deep_net()

compare(50)

# model_best = build_net()
# plot_best = train_net(model_best, 50
#                       datagen=data,
#                       testgen=test,
#                       h5_path="cifar100_best.h5")
# train_acc = plot_best.history['acc']
# test_acc = plot_best.history['val_acc']
#
# print train_acc
# print test_acc

# Default

# Best
[0.15393999999999999, 0.2797, 0.33444000000000002, 0.36806, 0.39328000000000002, 0.41249999999999998, 0.42934, 0.44738, 0.46011999999999997, 0.47148000000000001, 0.48527999999999999, 0.49347999999999997, 0.49928, 0.51044, 0.51854, 0.52380000000000004, 0.53525999999999996, 0.53668000000000005, 0.54461999999999999, 0.54778000000000004, 0.55234000000000005, 0.55881999999999998, 0.56532000000000004, 0.56742000000000004, 0.57020000000000004, 0.57308000000000003, 0.57931999999999995, 0.5837, 0.58804000000000001, 0.58977999999999997, 0.59396000000000004, 0.59528000000000003, 0.59882000000000002, 0.60138000000000003, 0.60426000000000002, 0.60907999999999995, 0.61082000000000003, 0.61514000000000002, 0.61241999999999996, 0.61906000000000005, 0.62419999999999998, 0.62194000000000005, 0.62404000000000004, 0.62738000000000005, 0.62829999999999997, 0.63126000000000004, 0.63368000000000002, 0.63795999999999997, 0.63646000000000003, 0.63880000000000003]
[0.23269999999999999, 0.31190000000000001, 0.38900000000000001, 0.41310000000000002, 0.43580000000000002, 0.45989999999999998, 0.46860000000000002, 0.4874, 0.50919999999999999, 0.49569999999999997, 0.52810000000000001, 0.52790000000000004, 0.53739999999999999, 0.53690000000000004, 0.5464, 0.5454, 0.56089999999999995, 0.5444, 0.57310000000000005, 0.56559999999999999, 0.56930000000000003, 0.58550000000000002, 0.58709999999999996, 0.58530000000000004, 0.58040000000000003, 0.59179999999999999, 0.58909999999999996, 0.59019999999999995, 0.58350000000000002, 0.57930000000000004, 0.59650000000000003, 0.59730000000000005, 0.59830000000000005, 0.5958, 0.58509999999999995, 0.60360000000000003, 0.60560000000000003, 0.5988, 0.60529999999999995, 0.6159, 0.59640000000000004, 0.60799999999999998, 0.60450000000000004, 0.6119, 0.60670000000000002, 0.59960000000000002, 0.60609999999999997, 0.61170000000000002, 0.6048, 0.60809999999999997]

# 100
[0.112, 0.21260000000000001, 0.27660000000000001, 0.3256, 0.37109999999999999, 0.40939999999999999, 0.43769999999999998, 0.4788, 0.50019999999999998, 0.5282, 0.55159999999999998, 0.57969999999999999, 0.60929999999999995, 0.63639999999999997, 0.6512, 0.67779999999999996, 0.70250000000000001, 0.71930000000000005, 0.72789999999999999, 0.75260000000000005, 0.77170000000000005, 0.78180000000000005, 0.80479999999999996, 0.81120000000000003, 0.81930000000000003, 0.82150000000000001, 0.84230000000000005, 0.84079999999999999, 0.85550000000000004, 0.85309999999999997, 0.87209999999999999, 0.87639999999999996, 0.87319999999999998, 0.87860000000000005, 0.88129999999999997, 0.88449999999999995, 0.89080000000000004, 0.9012, 0.89859999999999995, 0.90569999999999995, 0.90469999999999995, 0.90569999999999995, 0.91069999999999995, 0.91500000000000004, 0.91420000000000001, 0.91500000000000004, 0.91920000000000002, 0.92320000000000002, 0.92400000000000004, 0.92359999999999998]
[0.1153, 0.1678, 0.23280000000000001, 0.2797, 0.27579999999999999, 0.29189999999999999, 0.29949999999999999, 0.28589999999999999, 0.31369999999999998, 0.35580000000000001, 0.33129999999999998, 0.3543, 0.36080000000000001, 0.36620000000000003, 0.36799999999999999, 0.37090000000000001, 0.39040000000000002, 0.36799999999999999, 0.3826, 0.3614, 0.37930000000000003, 0.3846, 0.38869999999999999, 0.38159999999999999, 0.39019999999999999, 0.38779999999999998, 0.39600000000000002, 0.38569999999999999, 0.38540000000000002, 0.38950000000000001, 0.39340000000000003, 0.39379999999999998, 0.38240000000000002, 0.39389999999999997, 0.37830000000000003, 0.38579999999999998, 0.39090000000000003, 0.38109999999999999, 0.39779999999999999, 0.38800000000000001, 0.37909999999999999, 0.40229999999999999, 0.40560000000000002, 0.38779999999999998, 0.3997, 0.38250000000000001, 0.38529999999999998, 0.39929999999999999, 0.38540000000000002, 0.4073]

# 300
[0.18790000000000001, 0.32200000000000001, 0.38033333333333336, 0.41973333333333335, 0.45643333333333336, 0.48126666666666668, 0.50936666666666663, 0.53273333333333328, 0.55636666666666668, 0.5803666666666667, 0.5907, 0.61053333333333337, 0.62429999999999997, 0.64676666666666671, 0.65443333333333331, 0.66896666666666671, 0.67996666666666672, 0.69499999999999995, 0.70586666666666664, 0.71636666666666671, 0.72919999999999996, 0.73383333333333334, 0.74870000000000003, 0.75666666666666671, 0.7624333333333333, 0.77306666666666668, 0.77573333333333339, 0.78176666666666672, 0.79473333333333329, 0.79843333333333333, 0.80506666666666671, 0.8034, 0.81366666666666665, 0.82013333333333338, 0.82163333333333333, 0.82903333333333329, 0.82940000000000003, 0.8370333333333333, 0.83999999999999997, 0.84516666666666662, 0.8440333333333333, 0.84756666666666669, 0.85009999999999997, 0.85670000000000002, 0.8605666666666667, 0.85766666666666669, 0.86283333333333334, 0.86886666666666668, 0.86990000000000001, 0.87150000000000005]
[0.24660000000000001, 0.32119999999999999, 0.39129999999999998, 0.3407, 0.41339999999999999, 0.41099999999999998, 0.41439999999999999, 0.45540000000000003, 0.4536, 0.46879999999999999, 0.48830000000000001, 0.48409999999999997, 0.47620000000000001, 0.50119999999999998, 0.50219999999999998, 0.50439999999999996, 0.48559999999999998, 0.51619999999999999, 0.52070000000000005, 0.52339999999999998, 0.51739999999999997, 0.51839999999999997, 0.52329999999999999, 0.52410000000000001, 0.51229999999999998, 0.5242, 0.5333, 0.51849999999999996, 0.53520000000000001, 0.52769999999999995, 0.52159999999999995, 0.52539999999999998, 0.52849999999999997, 0.52739999999999998, 0.52559999999999996, 0.52000000000000002, 0.52710000000000001, 0.52990000000000004, 0.52329999999999999, 0.5323, 0.52249999999999996, 0.52349999999999997, 0.52790000000000004, 0.5333, 0.53549999999999998, 0.51500000000000001, 0.52380000000000004, 0.53280000000000005, 0.53800000000000003, 0.52910000000000001]

# Shallow