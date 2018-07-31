

import keras
import keras.backend as K
import numpy as np

from utils import anchors as utils_anchors

class Anchors(keras.layers.Layer):

    def __init__(self, size, stride, ratios=None, scales=None, *args, **keywords):
        #super().__init__(*args, **keywords)
        super().__init__(*args, **keywords)
        self.size = size
        self.stride = stride
        self.ratios = ratios
        self.scales = scales

        if ratios is None:
            #"""As in [20], at each pyramid level we use anchors at
            #three aspect ratios {1:2, 1:1, 2:1}"""
            self.ratios = np.array([0.5, 1, 2])
    
        if scales is None:
            #"""For denser scale coverage than in [20], at each level 
            #we add anchors of sizes {2 0 ,2 1/3 , 2 2/3 } of the original 
            #set of 3 aspect ratio anchors."""
            self.scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

        self.num_anchors = len(self.ratios) * len(self.scales)
        self.anchors = K.variable(utils_anchors.generate_anchors(
            base_size=self.size,
            ratios=self.ratios,
            scales=self.scales
        ))

    def call(self, inputs, **kwargs):
        features = inputs
        features_shape = K.shape(features)[:3]
    
        anchors = utils_anchors.shift(features_shape[1:3], self.stride, self.anchors)
        anchors = K.tile(K.expand_dims(anchors, axis=0), (features_shape[0], 1, 1))

        return anchors
    
    def compute_output_shape(self, input_shape):
        if None not in input_shape[1:]:
            total = np.prod(input_shape[1:3]) * self.num_anchors
            return (input_shape[0], total, 4)

        else:
            return (input_shape[0], None, 4)
        
    def get_config(self):
        config = super().get_config()

        #deep copy
        config.update({
            'size' : self.size,
            'stride' : self.stride,
            'ratios' : self.ratios.tolist(),
            'scales' : self.scales.tolist(),
        })

        return config

#a = Anchors(1, 1, name='ff')
#print(a.get_config())
#a.name='ff'
##c = keras.layers.Conv2D(11, 3, name='ff')
#print(a(K.variable(np.random.rand(2, 5, 5, 3))))