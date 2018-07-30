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

def default_submodels():

    return [
        ('regression', box_regression_subnet()),
        ('classification', classification_subnet())
    ]
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
        'filters' : feature_size,
        'strides' : 1,
        'padding' : 'same',
        'use_bias' : False
    }

    #bottom-down path

    #perform 1x1 conv
    M5 = keras.layers.Conv2D(kernel_size=1, name='C5_1x1', **options)(C5)
    M4 = keras.layers.Conv2D(kernel_size=1, name='C4_1x1', **options)(C4)
    M3 = keras.layers.Conv2D(kernel_size=1, name='C3_1x1', **options)(C3)

    M5_upsampled = layers.Upsamplingx2(name='M5_x2')(M5)
    #M5_upsampled = layers.UpsamplingSame(name='M5_x2')([M5, M4])
    M4 = keras.layers.Add(name='M5_upsampled_M4_merge')([M5_upsampled, M4])

    M4_upsampled = layers.Upsamplingx2()(M4)
    #M4_upsampled = layers.UpsamplingSame()([M4, M3])
    M3 = keras.layers.Add(name='M4_upsampled_M3_merge')([M4_upsampled, M3])

    #get P5
    P5 = keras.layers.Conv2D(kernel_size=3, name='M5_3x3', **options)(M5)
    P4 = keras.layers.Conv2D(kernel_size=3, name='M4_3x3', **options)(M4)
    P3 = keras.layers.Conv2D(kernel_size=3, name='M3_3x3', **options)(M3)

    #"""P6 is obtained via a 3×3 stride-2 conv on C5"""
    P6 = keras.layers.Conv2D(
        feature_size, 
        kernel_size=3, 
        strides=2, 
        padding='same',
        use_bias=False,
        name='P6')(C5)

    #"""and P7 is computed by applying ReLU followed by a 3×3 stride-2 conv on P6."""
    P7 = keras.layers.Activation('relu', name='P7_relu')(P6)
    P7 = keras.layers.Conv2D(
        feature_size,
        kernel_size=3,
        strides=2,
        padding='same',
        use_bias=False,
        name='P7')(P7)

    return [P3, P4, P5, P6, P7]

class AnchorParameters:
    
    def __init__(self, sizes, strides, ratios, scales):
        self.sizes = sizes
        self.strides = strides
        self.ratios = ratios
        self.scales = scales
    
    def num_anchors(self):
        return len(self.ratios) * len(self.scales)

def _build_model_pyramid(name, model, features):
    """concatenate subnet response for each kinds of subnet

    Args:
        name: output tensor name
        model: subnet 
        features: FPN features [P3, P4, P5, P6, P7]
    
    Returns:
        concantenated tensor
    """

    return keras.layers.Concatenate(axis=1, name=name)([model(f) for f in features])

def _build_pyramid(models, features):
    """Apllies all submodels to each features

    Args:
        models: two subnets
        features: [P3, P4, P5, P6, P7]
    
    Returnes:
        2 tensors: concatenated regression subnet tensor and classification
                   subnet tensor
    """

    return [_build_model_pyramid(n, m, features) for n, m in models]

def _build_anchors(anchor_parameters, features):
    
def retinanet(
    inputs,
    backbone_layers,
    num_classes=settings.K,
    num_anchors=settings.A,
    create_pyramid_features=_create_pyramid_features,
    submodels=None,
    name='retinanet'):

    if submodels is None:
        submodels = default_submodels()

    features = create_pyramid_features(K.variable(C1), K.variable(C2), K.variable(C3))

    #"""The object classification subnet and the box regression subnet, 
    #though sharing a common structure, use separate parameters."""
    pyramids = _build_pyramid(submodels, features)
    for p in pyramids:
        print(p)

subnet = box_regression_subnet()
import numpy as np

C3 = np.random.random((100, 16, 16, 256))
C2 = np.random.random((100, 32, 32, 256))
C1 = np.random.random((100, 64, 64, 256))


y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)

from keras import backend as K

#_create_pyramid_features(K.variable(C1), K.variable(C2), K.variable(C3))
import cProfile
#retinanet(1, 2)
cProfile.runctx('retinanet(1, 2)', globals(), None)
#print(subnet.summary())

