"""
"""
from .vgg import VGG16
from .mobilenet_v2 import MobileNetV2


def create_model(cfg):
    """

    Args
      :cfg:

    Returns
      :model: An instance of the model.
    """
    backbone = cfg['MODEL']['NAME']
    image_size = cfg['INPUT']['IMAGE_SIZE']
    num_classes = cfg['MODEL']['NUM_CLASSES']
    boxes_per_location = cfg['MODEL']['PRIORS']['BOXES_PER_LOCATION']
    input_shape = (image_size, image_size, 3)

    if backbone == 'VGG16':
        model = VGG16(
            input_shape, num_classes, boxes_per_location, name='VGG16')
    elif backbone == 'MobileNetV2':
        model = MobileNetV2(input_shape, num_classes,
                            boxes_per_location, 'MobileNetV2_SSD')
    else:
        raise ValueError(f'Not supported backbone: {backbone}')
    return model
