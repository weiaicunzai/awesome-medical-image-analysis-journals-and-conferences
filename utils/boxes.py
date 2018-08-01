

import keras
import keras.backend as K

def bbox_transform_inv(boxes, deltas, mean=None, std=None):



    if mean is None:
        mean = [0, 0, 0, 0]
    
    if std is None:
        std = [0.2, 0.2, 0.2, 0.2]
    
    width = boxes[:, :, 2] - boxes[:, :, 0]
    height = boxes[:, :, 3] - boxes[:, :, 1]

    x1 = boxes[:, :, 0] + (deltas[:, :, 0] * std[0] + mean[0]) * width
    y1 = boxes[:, :, 1] + (deltas[:, :, 1] * std[1] + mean[1]) * height
    x2 = boxes[:, :, 2] + (deltas[:, :, 2] * std[2] + mean[2]) * width
    y2 = boxes[:, :, 3] + (deltas[:, :, 3] * std[3] + mean[3]) * height

    pred_boxes = K.stack([x1, x2, y1, y2], axis=2)

    return pred_boxes