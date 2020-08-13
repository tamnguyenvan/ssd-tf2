"""
"""
import tensorflow as tf


def pad_to_fixed_size(input_tensor, size, values=0):
    """
    """
    input_shape = input_tensor.get_shape().as_list()
    padding_shape = []

    # Computes the padding length on the first dimension.
    padding_length = tf.maximum(0, size - tf.shape(input_tensor)[0])
    assert_length = tf.Assert(
    tf.greater_equal(padding_length, 0), [padding_length])
    with tf.control_dependencies([assert_length]):
        padding_shape.append(padding_length)

    # Copies shapes of the rest of input shape dimensions.
    for i in range(1, len(input_shape)):
        padding_shape.append(tf.shape(input=input_tensor)[i])

    # Pads input tensor to the fixed first dimension.
    paddings = tf.cast(values * tf.ones(padding_shape),
                       input_tensor.dtype)
    padded_tensor = tf.concat([input_tensor, paddings], axis=0)[:size, ...]
    output_shape = input_shape
    output_shape[0] = size
    padded_tensor.set_shape(output_shape)
    return padded_tensor

