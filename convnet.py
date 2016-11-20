import cPickle, gzip
import numpy as np
import math
#import matplotlib.pyplot as plt
from keras.datasets import cifar100, cifar10
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Flatten, SpatialDropout2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils.np_utils import to_categorical
from keras.callbacks import Callback
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
#from scipy.misc import imsave

def load_data(dataset, images_per_cat=None):

    if dataset == "cifar100":
        data = cifar100.load_data()
    elif dataset =="cifar10":
        data = cifar10.load_data()

    train_data = data[0][0]
    train_labels = data[0][1]
    test_data = data[1][0]
    test_labels = data[1][1]

    if images_per_cat:
        num_cat = max(train_labels) + 1
        
        for c in range(num_cat):
            T_cat = T == c
            ind = np.nonzero(T_cat)

    return train_data, train_labels, test_data, test_labels

class TestLoss(Callback):
    def __init__(self, test_data, test_plot,model):
        super(Callback,self).__init__()
        self.test_data = test_data
        self.model = model
        self.test_plot = test_plot

    def on_epoch_end(self, batch, logs={}):
        X_test, T_test_onehot = self.test_data
        print "Test"
        loss, acc = self.model.evaluate(X_test,T_test_onehot, batch_size=32)
        print "loss:",loss,"Acc:",acc
        self.test_plot.append((loss, acc))
"""
def show_image(ind):
    image = train_data[ind,:,:,:]
    plt.imshow(image)
    plt.show()
""" 
def build_net():
    (X,T,X_test,T_test) = load_data("cifar100")

   # X = X[0 : 2000,:]
   # T = T[0 : 2000, :]
    X = X.astype(float)
    X_test = X_test.astype(float)

    T_onehot = to_categorical(T,100)
    T_test_onehot = to_categorical(T_test,100)

    datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True, horizontal_flip=True, width_shift_range=0.1, height_shift_range=0.1)
    datagen_test = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)# horizontal_flip=True)

    datagen.fit(X)
    datagen_test.fit(X_test)

    model = Sequential()
    # model.add(BatchNormalization(input_shape=(32, 32, 3)))

    #ConvLayer 1
    model.add(Convolution2D(32, 3, 3, init='normal', input_shape=(32, 32, 3)))
    model.add(BatchNormalization(mode=0, axis=1))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    #ConvLayer 2
    #model.add(Convolution2D(64, 3, 3, init='normal'))
    #model.add(BatchNormalization(mode=0, axis=1))
    #model.add(Activation("relu"))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(SpatialDropout2D(0.15))
    
    #ConvLayer 3
    #model.add(Convolution2D(128, 3, 3, init='normal'))
    #model.add(BatchNormalization(mode=0, axis=1))
    #model.add(Activation("relu"))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(SpatialDropout2D(0.2))

    #ConvLayer 4
    #model.add(Convolution2D(128,3,3, init='normal'))
    #model.add(BatchNormalization(mode=0,axis=1))
    #model.add(Activation("relu"))
    #model.add(SpatialDropout2D(0.2))
    
    #Fully connected layer
    model.add(Flatten())
    model.add(Dense(512, init='normal'))
    model.add(BatchNormalization(mode=0, axis=1))
    model.add(Activation("relu"))
    model.add(Dropout(0.1))

    #Output Softmax Layer
    model.add(Dense(100, init='normal', activation="softmax"))

    model.summary()
 
    #Train

    model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    
    #test_plot = []
    #train_plot = model.fit(X,T_onehot,nb_epoch=100, batch_size=32, validation_data=(X_test,T_test_onehot))#callbacks=[TestLoss((X_test,T_test_onehot), test_plot, model)],batch_size=32)

    train_plot =  model.fit_generator(datagen.flow(X,T_onehot, batch_size=32), validation_data=datagen_test.flow(X_test,T_test_onehot),nb_val_samples=len(X_test), nb_epoch=5, samples_per_epoch=(len(X)))

    model.save("cifar100_model.h5")

    #print train_plot.history["val_acc"]
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

def fine_tuning_cifar10():
    (X,T,X_test,T_test) = load_data("cifar10")

    X = X.astype(float)
    X_test = X_test.astype(float)

    T_onehot = to_categorical(T,10)
    T_test_onehot = to_categorical(T_test,10)

    # Data Preprocesing
    datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True, horizontal_flip=True, width_shift_range=0.1, height_shift_range=0.1)
    datagen_test = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)# horizontal_flip=True)

    datagen.fit(X)
    datagen_test.fit(X_test)
   
    # Cifar100 model delete last layer and set the first layers to be non-trainable
    model = load_model("cifar100_model.h5")
    model.pop()
    for l in model.layers:
        if l.name == "convolution2d_3":
            break
        l.trainable = False

    # Model built on top of Cifar100
    model_top = Sequential()
    model_top.add(Dense(10, init='normal', activation="softmax", input_shape=(512,)))
    
    model.add(model_top)
    
    #Train

    model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    
    test_plot = []

    train_plot =  model.fit_generator(datagen.flow(X,T_onehot, batch_size=32), validation_data=datagen_test.flow(X_test,T_test_onehot),nb_val_samples=len(X_test), nb_epoch=100, samples_per_epoch=(len(X)))
   
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def visualize_filters():
    layer_name = "convolution2d_1"

    filter_index = 0
    model = load_model("cifar100_model.h5")

    layer_dict = dict([(layer.name, layer) for layer in model.layers])

    layer_output = layer_dict[layer_name].output
    loss = K.mean(layer_output[:, filter_index, :, :])

    # compute the gradient of the input picture wrt this loss
    grads = K.gradients(loss, input_img)[0]

    # normalization trick: we normalize the gradient
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

    # this function returns the loss and grads given the input picture
    iterate = K.function([input_img], [loss, grads])

    input_img_data = np.random.random((1, 3, 32, 32)) * 20 + 32.

    for i in range(20):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step

    img = input_img_data[0]
    img = deprocess_image(img)

build_net()    
fine_tuning_cifar10()
