"""Keras layer for upsample a feature shape
to another feature using nearest neighbors
"""

import keras

class Upsamplingx2(keras.layers.Layer):

    #use bilinear interpolation instead of nearest neighbor
    def call(self, inputs):
        return keras.backend.resize_images(inputs, 2, 2, 'channels_last')
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] * 2, input_shape[2] * 2, input_shape[3])
        
class UpsamplingSame(keras.layers.Layer):

    #use deconvlotion to upsample
    def call(self, inputs):
        source, target = inputs
        self.target_shape = keras.backend.int_shape(target)
        source = keras.layers.Conv2DTranspose(
            self.target_shape[-1],
            kernel_size=3,
            strides=2,
            padding='valid')(source)
        
        source = keras.layers.Lambda(
            lambda x: x[:self.target_shape[1], :self.target_shape[2]])(source)

        return source
    
    def compute_output_shape(self, input_shape):
        return self.target_shape
