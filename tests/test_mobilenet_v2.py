import context

import tensorflow as tf
from models import mobilenet_v2


physical_devices = tf.config.experimental.list_physical_devices('GPU')
for physical_device in physical_devices:
    tf.config.experimental.set_memory_growth(physical_device, True)

model = mobilenet_v2.MobileNetV2([320, 320, 3], 81, 'ssd_mobilenet_v2')
model.summary()

x = tf.random.normal((1, 320, 320, 3))
outs = model(x)
for o in outs:
    print(o.shape)
