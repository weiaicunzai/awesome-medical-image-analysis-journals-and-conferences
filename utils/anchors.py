


import numpy as np
import keras

def generate_anchors(base_size=16, ratios=None, scales=None):

    if ratios is None:
        #"""As in [20], at each pyramid level we use anchors at
        #three aspect ratios {1:2, 1:1, 2:1}"""
        ratios = np.array([0.5, 1, 2])
    
    if scales is None:
        #"""For denser scale coverage than in [20], at each level 
        #we add anchors of sizes {2 0 ,2 1/3 , 2 2/3 } of the original 
        #set of 3 aspect ratio anchors."""
        scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])
    
    num_anchors = len(ratios) * len(scales)

    #initialize output anchors
    anchors = np.zeros((num_anchors, 4))

    anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T

    areas = anchors[:, 2] * anchors[:, 3]

    #ratio = w / h
    #area = w * h
    #area * ratio = area / w * h = w * h / w * h = h^2
    #h * ratio = h * w / h = w
    anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
    anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))

    # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

    return anchors


    




    