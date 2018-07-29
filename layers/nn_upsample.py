"""Keras layer for upsample a feature shape
to another feature using nearest neighbors
"""

import keras

class NearestNeighborUpsampling(keras.layers.Layer):

    #use bilinear interpolation instead of nearest neighbor
    def call(self, inputs):
        source, target = inputs
        print('source: ', keras.backend.int_shape(source))
        print('targets: ', keras.backend.int_shape(target))
        target_shape = keras.backend.int_shape(target)
        #print(type(target_shape[1]))
        return keras.backend.resize_images(source, target_shape[1], target_shape[2], 'channels_last')
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[1][1], input_shape[1][2], input_shape[0][-1])
        