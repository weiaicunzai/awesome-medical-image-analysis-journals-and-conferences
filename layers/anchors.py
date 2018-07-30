

import keras
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
        self.anchors = keras.backend.variable(utils_anchors.generate_anchors(
            base_size=self.size,
            ratios=self.ratios,
            scales=self.scales
        ))

    def call(self, inputs, **kwargs):
        features = inputs
        features_shape = keras.backend.
a = Anchors(1, 1, name='ff')
a.name='ff'
#c = keras.layers.Conv2D(11, 3, name='ff')
print(a)