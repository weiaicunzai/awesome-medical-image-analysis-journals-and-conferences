

import keras
import keras.backend as K

class ClipBoxes(keras.layers.Layer):

    """clip bounding box edges which out of
    image shape"""

    def call(self, inputs, **kwargs):
        image, boxes = inputs
        shape = K.cast(K.shape(image), K.floatx())

        x1 = K.clip(boxes[:, :, 0], 0, K.eval(shape[2]))
        x2 = K.clip(boxes[:, :, 1], 0, K.eval(shape[1]))
        y1 = K.clip(boxes[:, :, 2], 0, K.eval(shape[2]))
        y2 = K.clip(boxes[:, :, 3], 0, K.eval(shape[1]))

        return K.stack([x1, y1, x2, y2], axis=2)
    
    def compute_output_shape(self, input_shape):
        return input_shape[1]
