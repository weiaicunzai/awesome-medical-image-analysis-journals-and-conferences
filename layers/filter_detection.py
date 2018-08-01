

import keras
import keras.backend as K
def filter_detection(
    boxes,
    classification,
    class_specific_filter=True,
    nms=True,
    score_threshold=0.05,
    max_detection=300,
    nms_threshold=0.5):

    def _filter_detections(scores, labels):
        #threshold based on score
        #indices = 
        import numpy as np
        a = K.greater(K.variable(np.random.randn(3,3), K.variable(np.random.rand(3, 3))))
        
        print(K.eval(a))

import numpy as np
test = K.variable(np.random.randn(3, 3))
a = K.greater(K.variable(np.random.randn(3,3)), K.variable(np.random.rand(3, 3)))

#b = test[a]
import tensorflow as tf
b = tf.where(a)
print(K.eval(a))
print(K.eval(b))