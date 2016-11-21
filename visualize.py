import numpy as np
from keras import backend as K
from keras.layers import Dense
from keras.models import load_model
from scipy.misc import imsave

def visualize(layer_dict, layer_name, step):
    layer = layer_dict[layer_name]
    layer_input = layer.input
    # layer.nb_filter

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
    x = np.clip(x, 0, 255).astype('uint8')
    return x

model = load_model("cifar100_best.h5")
layer_dict = {layer.name: layer for layer in model.layers}
#
# layer_name = 'convolution2d_1'
# layer_name = 'convolution2d_2'
# input_img = model.layers[0].input
# layer_output = layer_dict[layer_name].output
#
# filter_index = 0
# loss = K.mean(layer_output[:, filter_index, :, :])
# grads = K.gradients(loss, input_img)[0]
# grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
# iterate = K.function([input_img], [loss, grads])
#
# step = 0.1
# input_img_data = np.random.random((1, 32, 32, 3)) * 20 + 128.
# for i in range(20):
#     loss_value, grads_value = iterate([input_img_data])
#     input_img_data += grads_value * step
#
# img = input_img_data[0]
# img = deprocess_image(img)
# imsave('%s_filter_%d.png' % (layer_name, filter_index), img)