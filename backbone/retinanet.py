"""
 RetinaNet Keras implementation:
 Focal Loss for Dense Object Detection
 Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollár
 https://arxiv.org/abs/1708.02002v2

Author:baiyu
"""

import keras
import layers

from config import settings

#"""We use C = 256 and A = 9 in most experiments."""
def classification_subnet(
    num_classes = settings.K,
    num_anchors=settings.A,
    pryamid_feature_size=settings.C,
    prior_probability=0.01,
    classification_feature_size=settings.C,
    name='classification_subnet'):
    """clssification_subnet

    Args:
        num_classes: Number of classes to predict a score for at each feature level.
        num_anchors: Number of anchors to predict clssification scores for each feature level 
        pyramid_feature_size: The number of filters to expect from the feature pyramid levels
        classification_feature_size: name of this subnet

    Returns:
        A keras.models.Model that predicts classes for each anchor
    """

    options = {
        'kernel_size' : 3,
        'strides' : 1,
        'padding' : 'same',
        'kernel_initializer' : 'he_normal',
        'use_bias': False
    }

    subnet = keras.models.Sequential()
    subnet.add(keras.layers.InputLayer(
        input_shape=(None, None, pryamid_feature_size),
        name='classification_subnet_input'
    ))
    
    #"""Taking an input feature map with C channels
    #from a given pyramid level, the subnet applies four 3×3
    #conv layers, each with C filters and each followed by ReLU
    #activations, followed by a 3×3 conv layer with KA filters."""
    for i in range(4):
        subnet.add(keras.layers.Conv2D(
            filters=classification_feature_size,
            name='classification_subnet_conv{}'.format(i),
            **options
        ))
        subnet.add(keras.layers.BatchNormalization(
            name='classification_subnet_bn{}'.format(i)
        ))
        subnet.add(keras.layers.Activation(
            'relu',
            name='classification_subnet_relu{}'.format(i)
            ))
    subnet.add(keras.layers.Conv2D(
        filters=num_anchors * num_classes,
        **options
    ))

    #"""Finally sigmoid activations are attached to output the KA
    #binary predictions per spatial location, see Figure 3 (c)."""

    subnet.add(keras.layers.Reshape(
        (-1, num_classes),
        name='pyramid_classification_reshape'
    ))
    subnet.add(keras.layers.Activation(
        'sigmoid',
        name='pyramid_classification_sigmoid'
    ))

    subnet.name = name
    return subnet

def box_regression_subnet(
    num_anchors=settings.A,
    pyramid_feature_size=settings.C,
    regression_feature_size=settings.C,
    name='regression_subnet'):

    """box_regression_subnet

    Args:
        num_anchors: Number of anchors for each feature level
        pyramid_feauture_size: the number of filters to expect from pyramid features
        regression_feature_size: the number of filters regression subnet use per layer
        name: name of the subnet
    
    Returns:
        a subnet Sequential() object
    """
    options = {
        'kernel_size' : 3,
        'strides' : 1,
        'padding' : 'same',
        'kernel_initializer' : 'he_normal',
        'use_bias' : False
    }

    subnet = keras.models.Sequential()
    subnet.add(keras.layers.InputLayer(input_shape=(None, None, pyramid_feature_size)))

    #'''The design of the box regression subnet is identical to the
    #classification subnet except that it terminates in 4A linear
    #outputs per spatial location, see Figure 3 (d).'''
    for i in range(4):
        subnet.add(keras.layers.Conv2D(
            filters=regression_feature_size,
            name='regression_subnet_conv{}'.format(i),
            **options
        ))
        subnet.add(keras.layers.BatchNormalization(
            name='regression_subnet_bn{}'.format(i)
        ))
        subnet.add(keras.layers.Activation(
            "relu",
            name='regression_subnet_{}'.format(i)
        ))

    subnet.add(keras.layers.Conv2D(
        num_anchors * 4,
        name='pyramid_regression_{}'.format(i),
        **options
    ))
    subnet.add(keras.layers.Reshape(
        (-1, 4),
        name='pyramid_regression_reshape'
    ))

    subnet.name = 'box_regression_subnet'
    return subnet

def _create_pyramid_features(C3, C4, C5, feature_size=settings.C):
    """ Create FPN layers on top of the backbone features

    Args:
        C3: Feature stage of C3 from backbone
        C4: Feature stage of C4 from backbone
        C5: Featrue stage of C5 from backbone
    
    Retruens:
        A list of feature level
    """
    options = {
        'filter_size' : feature_size,
        'strides' : 1,
        'padding' : 'same'
    }

    M5 = keras.layers.Conv2D(kernel_size=1, name='C5_reduced', **options)(C5)
    P5 = keras.layers.Conv2D(kernel_size=3, name='P5', **options)(M5)
    M5_unsampled = layers.NearestNeighborUpsampling()[M5, M4]
    M4 = keras.layers.Conv2D(
        kernel_size=1,
        name='C4_reduced',
        **options
    )(C4)
    M4 = keras.layers.Add()([M4, M5_unsampled])

    M3 = keras.layers.Conv2D(
        kernel_size=1,
        name='C4_reduced',
        **options
    )(C3)






    P4_unsampled = layers.NearestNeighborUpsampling()[]
    
    P5 = layers.NearestNeighborUpsampling()([M5, C4])
    

#def retinanet(
#
#)
#subnet = classification_subnet(40, 40)
subnet = box_regression_subnet()
import numpy as np

C1 = np.random.random((100, 16, 16, 256))
C2 = np.random.random((100, 32, 32, 256))
C3 = np.random.random((100, 32, 32, 256))


y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)

from keras import backend as K

_create_pyramid_features(K.variable(C1), K.variable(C2), K.variable(C3))
import cProfile
#cProfile.runctx('box_regression_subnet()', globals(), None)
#print(subnet.summary())

