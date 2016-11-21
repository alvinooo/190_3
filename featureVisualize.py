from __future__ import print_function
from scipy.misc import imsave
from keras.models import load_model
import numpy as np
import time
from keras.applications import vgg16
from keras import backend as K
import sys

# dimensions of the generated pictures for each filter.
img_width = 32 
img_height = 32
filters_layer1 = []
filters_layer2 = []

n1 = int(sys.argv[1])
n2 = int(sys.argv[2]) 

model = load_model("cifar100_deep.h5")
# this is the placeholder for the input images
input_img = model.input

# get the symbolic outputs of each "key" layer (we gave them unique names).
layer_dict = dict([(layer.name, layer) for layer in model.layers[0:]])

print(layer_dict)

# util function to convert a tensor into a valid image
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
    if K.image_dim_ordering() == 'th':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def compute_gradient_ascent(layer_name, num_filters):

    for filter_index in range(0, num_filters):
    # we only scan through the first 200 filters,
    # but there are actually 512 of them
    #rint('Processing filter %d' % filter_index)

    # we build a loss function that maximizes the activation
    # of the nth filter of the layer considered
        layer_output = layer_dict[layer_name].output
        if K.image_dim_ordering() == 'th':
            loss = K.mean(layer_output[:, filter_index, :, :])
        else:
            loss = K.mean(layer_output[:, :, :, filter_index])

    # we compute the gradient of the input picture wrt this loss
        grads = K.gradients(loss, input_img)[0]

    # normalization trick: we normalize the gradient
        grads = normalize(grads)

    # this function returns the loss and grads given the input picture
        iterate = K.function([input_img, K.learning_phase()], [loss, grads])
    # step size for gradient ascent
        step = 1.

    # we start from a gray image with some random noise
        if K.image_dim_ordering() == 'th':
            input_img_data = np.random.random((1, 3, img_width, img_height))
        else:
            input_img_data = np.random.random((1, img_width, img_height, 3))
        input_img_data = (input_img_data - 0.5) * 20 + 128

    # we run gradient ascent for 20 steps
        for i in range(20):
            loss_value, grads_value = iterate([input_img_data, 0] )
            input_img_data += grads_value * step

            #print('Current loss value:', loss_value)
            if loss_value <= 0.:
            # some filters get stuck to 0, we can skip them
                break

    # decode the resulting input image
        #if loss_value > 0:
        img = deprocess_image(input_img_data[0])
        if(layer_name == "convolution2d_1"):
            filters_layer1.append(img)
        #imsave('%s_filter_%d.png' % (layer_name, filter_index), img)
        if(layer_name == "convolution2d_2"):
            filters_layer2.append(img)

    
def feauture_visualize_layer():
    
    layer_name_1 = "convolution2d_1"
      
    compute_gradient_ascent(layer_name_1, 32 )
    
    layer_name_2 = "convolution2d_2"

    compute_gradient_ascent(layer_name_2, 64)

    for i in range((n1)):
        imsave('%s_filter_%d.png' % (layer_name_1, i), filters_layer1[i])

    for i in range((n2)):
        imsave('%s_filter_%d.png' % (layer_name_2, i), filters_layer2[i])

feauture_visualize_layer()

