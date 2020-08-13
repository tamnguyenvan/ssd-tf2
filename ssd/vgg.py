"""
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


def create_predictions(num_classes, boxes_per_location):
    """Create prediction heads.

    Args
      :num_classes: Number of classes.
      :boxes_per_location: Number of prior boxes per location.

    Returns
      :conf_layers: List of conv heads for conf prediction.
      :loc_layers: List of conv heads for location prediction.
    """
    conf_layers = []
    loc_layers = []
    for num_priors in boxes_per_location:
        conf_layers.append(
            layers.Conv2D(
                num_priors * num_classes, kernel_size=3, padding='same')
        )
        loc_layers.append(
            layers.Conv2D(num_priors * 4, kernel_size=3, padding='same')
        )

    return conf_layers, loc_layers


def create_custom_vgg16(input_shape, weights=None, name=None):
    """Create a custom VGG16 model.

    Args
      :input_shape: The input shape.
      :weights: The pretrained weights.
      :name: Model name.

    Returns
      :custom_model: A Functional object model.
    """
    custom_model = tf.keras.Sequential([
        layers.Input(input_shape, name='input_1'),
        layers.Conv2D(
            64, 3, padding='same', activation='relu', name='block1_conv1'),
        layers.Conv2D(
            64, 3, padding='same', activation='relu', name='block1_conv2'),
        layers.MaxPool2D(2, 2, padding='same', name='block1_pool'),

        layers.Conv2D(
            128, 3, padding='same', activation='relu', name='block2_conv1'),
        layers.Conv2D(
            128, 3, padding='same', activation='relu', name='block2_conv2'),
        layers.MaxPool2D(2, 2, padding='same', name='block2_pool'),

        layers.Conv2D(
            256, 3, padding='same', activation='relu', name='block3_conv1'),
        layers.Conv2D(
            256, 3, padding='same', activation='relu', name='block3_conv2'),
        layers.Conv2D(
            256, 3, padding='same', activation='relu', name='block3_conv3'),
        layers.MaxPool2D(2, 2, padding='same', name='block3_pool'),

        layers.Conv2D(
            512, 3, padding='same', activation='relu', name='block4_conv1'),
        layers.Conv2D(
            512, 3, padding='same', activation='relu', name='block4_conv2'),
        layers.Conv2D(
            512, 3, padding='same', activation='relu', name='block4_conv3'),
        layers.MaxPool2D(2, 2, padding='same', name='block4_pool'),

        layers.Conv2D(
            512, 3, padding='same', activation='relu', name='block5_conv1'),
        layers.Conv2D(
            512, 3, padding='same', activation='relu', name='block5_conv2'),
        layers.Conv2D(
            512, 3, padding='same', activation='relu', name='block5_conv3'),
    ], name=name)
    if weights:
        tf_vgg16 = tf.keras.applications.VGG16(
            input_shape=input_shape, include_top=False, weights=weights
        )
        for layer in tf_vgg16.layers[1:]:
            layer_name = layer.name
            if 'conv' not in layer_name:
                continue

            custom_layer = custom_model.get_layer(layer_name)
            if custom_layer is None:
                continue
            custom_layer.set_weights(layer.get_weights())
    return custom_model


def VGG16(input_shape=None, num_classes=None,
          boxes_per_location=[4, 6, 6, 6, 4, 4],
          weights='imagenet', name=None):
    """Instaniate a VGG-based SSD model.

    Args
      :input_shape: The input shape tuple.
      :num_classes: The number of classes.
      :weights: The pretrained weights.
      :name: The name of model.
    
    Returns
     :model: An Functional object model.
    """
    if not isinstance(num_classes, int):
        raise TypeError('Number of classes must be an integer')

    if num_classes <= 1:
        raise ValueError('Must be 2 classes at least')

    if len(boxes_per_location) != 6:
        raise ValueError('Must be 6 heads')

    base_model = create_custom_vgg16(input_shape, weights, name='vgg16_base')

    # The first features is conv4_3
    features = []
    conv4_3_out = base_model.get_layer('block4_conv3').output
    conv4_3_out_bn = layers.BatchNormalization()(conv4_3_out)
    features.append(conv4_3_out_bn)

    # Add 3 more layers
    conv5_3 = base_model.output
    pool5 = layers.MaxPooling2D(3, 1, padding='same')(conv5_3)
    conv6 = layers.Conv2D(1024, 3, padding='same', dilation_rate=6,
                          activation='relu', name='conv6')(pool5)
    conv7 = layers.Conv2D(
        1024, 1, padding='same', activation='relu', name='conv7')(conv6)
    features.append(conv7)

    # Add extra layers
    # 8th block output shape: B, 512, 10, 10
    x = layers.Conv2D(256, 1, activation='relu')(conv7)
    x = layers.Conv2D(512, 3, strides=2, padding='same', activation='relu')(x)
    features.append(x)

    # 9th block output shape: B, 256, 5, 5
    x = layers.Conv2D(128, 1, activation='relu')(x)
    x = layers.Conv2D(256, 3, strides=2, padding='same', activation='relu')(x)
    features.append(x)

    # 10th block output shape: B, 256, 3, 3
    x = layers.Conv2D(128, 1, activation='relu')(x)
    x = layers.Conv2D(256, 3, activation='relu')(x)
    features.append(x)

    # 11th block output shape: B, 256, 1, 1
    x = layers.Conv2D(128, 1, activation='relu')(x)
    x = layers.Conv2D(256, 3, activation='relu')(x)
    features.append(x)

    conf_layers, loc_layers = create_predictions(num_classes,
                                                 boxes_per_location)
    confs = []
    locs = []
    for x, c, l in zip(features, conf_layers, loc_layers):
        confs.append(c(x))
        locs.append(l(x))

    confs = [tf.reshape(o, (tf.shape(o)[0], -1, num_classes))
             for o in confs]
    locs = [tf.reshape(o, (tf.shape(o)[0], -1, 4))
            for o in locs]
    confs = tf.concat(confs, axis=1)
    locs = tf.concat(locs, axis=1)

    inputs = base_model.input
    outputs = (confs, locs)
    return tf.keras.Model(inputs, outputs, name=name)
