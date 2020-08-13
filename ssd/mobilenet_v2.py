"""
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .layers import Conv2DPadding, ConvBNReLU, inverted_resudial_block


BOXES_PER_LOCATION = [6, 6, 6, 6, 6, 6]


def MobileNetV2(input_shape=None, num_classes=None,
                boxes_per_location=BOXES_PER_LOCATION, name=None):
    """
    """
    if not isinstance(input_shape, (tuple, list)):
        raise ValueError('Input shape must be tuple or list')

    if len(input_shape) != 3:
        raise ValueError('Input shape must be 3-tuple or 3-list')

    if not isinstance(num_classes, int):
        raise ValueError('Number of classes must be an integer')

    origin_model = keras.applications.MobileNetV2(
        input_shape=input_shape, include_top=False
    )
    inputs = origin_model.input
    detections = []
    detections.append(origin_model.get_layer('block_14_add').output)
    x = origin_model.output
    detections.append(x)

    x = inverted_resudial_block(x, 1280, 512, 2, 0.2)
    detections.append(x)
    x = inverted_resudial_block(x, 512, 256, 2, 0.25)
    detections.append(x)
    x = inverted_resudial_block(x, 256, 256, 2, 0.5)
    detections.append(x)
    x = inverted_resudial_block(x, 256, 64, 2, 0.25)
    detections.append(x)

    confs = []
    locs = []
    for num_boxes, x in zip(BOXES_PER_LOCATION, detections):
        confs.append(Conv2DPadding(
            num_boxes * num_classes, kernel_size=3, stride=1, padding=1)(x))
        locs.append(Conv2DPadding(
            num_boxes * 4, kernel_size=3, stride=1, padding=1)(x))

    confs = [tf.reshape(o, (tf.shape(o)[0], -1, num_classes))
             for o in confs]
    locs = [tf.reshape(o, (tf.shape(o)[0], -1, 4))
            for o in locs]

    confs = tf.concat(confs, axis=1)
    locs = tf.concat(locs, axis=1)

    outputs = (confs, locs)
    return keras.Model(inputs, outputs, name=name)
