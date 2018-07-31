


import numpy as np
import keras
import keras.backend as K


def meshgrid(x, y):
    """simplfied np.meshgrid, only support 1D array

    Args:
        x: 1D tensor of x index
        y: 1D tensor of y index
    
    Returns:
        index matrix for x and y
    """
    #dont use int_shape, this could return a 
    #dynamic shape e.g.(None,) instead of
    #consice number during runtime
    if K.count_params(K.shape(x)) != 1:
        raise Exception('expected x to be 1D tensor')
    
    if K.count_params(K.shape(y)) != 1:
        raise Exception('expected y to be 1D tensor')
    
    x_num = K.shape(x)[0]
    y_num = K.shape(y)[0]

    x = K.expand_dims(x, axis=0)
    x = K.tile(x, (y_num, 1))

    y = K.transpose(y)
    y = K.expand_dims(y, axis=1)
    y = K.tile(y, (1, x_num))
    return x, y

def shift(shape, stride, anchors):
    """shift anchor positions

    Args:
        shape: feature map size to shift anchor over
        stride: stride to shift anchor
        anchors: the anthors to shift, default 9 anchors per feature map
    
    Returns:
        every possible location of shifted anchors
    """

    shift_x = (K.arange(0, shape[1], dtype=K.floatx()) + 
              K.constant(0.5, dtype=K.floatx())) * stride

    shift_y = (K.arange(0, shape[0], dtype=K.floatx()) +
               K.constant(0.5, dtype=K.floatx())) * stride

    x, y = meshgrid(shift_x, shift_y)
    shift_x = K.reshape(x, [-1])
    shift_y = K.reshape(y, [-1])

    shifts = K.stack([
        shift_x,
        shift_y,
        shift_x,
        shift_y
    ], axis=0)

    shifts = K.transpose(shifts)
    num_of_anchors = K.shape(anchors)[0]

    #how many shifts per anchor
    k = K.shape(shifts)[0]

    shifted_anchors = K.reshape(anchors, [1, num_of_anchors, 4]) + \
        K.cast(K.reshape(shifts, [k, 1, 4]), K.floatx())
    shifted_anchors = K.reshape(shifted_anchors, [k * num_of_anchors, 4])

    return shifted_anchors


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


    




    