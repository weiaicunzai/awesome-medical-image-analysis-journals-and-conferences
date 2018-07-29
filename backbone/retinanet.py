"""
 RetinaNet Keras implementation:
 Focal Loss for Dense Object Detection
 Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollár
 https://arxiv.org/abs/1708.02002v2

Author:baiyu
"""

import keras

#"""We use C = 256 and A = 9 in most experiments."""
def classification_subnet(num_classes, 
                        num_anchors=9,
                        pryamid_feature_size=256,
                        prior_probability=0.01,
                        classification_feature_size=256,
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
            name='pyramid_classification_{}'.format(i),
            **options
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

    subnet.name = 'classification_subnet'
    return subnet

subnet = classification_subnet(40, 40)

print(subnet.summary())

