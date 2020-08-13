"""
"""
from tensorflow.keras import layers


class Conv2DPadding(layers.Layer):
    def __init__(self, filters, kernel_size=3, stride=1, padding=0, **kwargs):
        super(Conv2DPadding, self).__init__()
        self.padding = layers.ZeroPadding2D(padding)
        self.conv = layers.Conv2D(
            filters, kernel_size, stride, padding='valid', **kwargs)

    def call(self, x):
        x = self.padding(x)
        x = self.conv(x)
        return x


class ConvBNReLU(layers.Layer):
    def __init__(self, filters, kernel_size=3, stride=1, groups=1):
        super(ConvBNReLU, self).__init__()

        padding = (kernel_size - 1) // 2
        self.padding = layers.ZeroPadding2D(padding)
        self.conv = layers.Conv2D(filters, kernel_size, stride,
                                  padding='valid', groups=groups,
                                  use_bias=False)
        self.bn = layers.BatchNormalization()
        self.relu = layers.Activation('relu')

    def call(self, x):
        x = self.padding(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


def inverted_resudial_block(x, filters_in, filters_out, stride, expand_ratio):
    """
    """
    inputs = x
    expand_dim = int(round(filters_in * expand_ratio))
    if expand_ratio > 1:
        x = ConvBNReLU(expand_dim, kernel_size=1)(x)
    x = ConvBNReLU(expand_dim, stride=stride, groups=expand_dim)(x)
    x = Conv2DPadding(filters_out, kernel_size=1, stride=1,
                      padding=0, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    if stride == 1 and filters_in == filters_out:
        return inputs + x
    return x


