import context

import yaml
import tensorflow as tf
from ssd import model_factory

devices = tf.config.experimental.list_physical_devices('GPU')
for device in devices:
    tf.config.experimental.set_memory_growth(device, True)

with open('configs/vgg_ssd300.yaml') as f:
    cfg = yaml.load(f, yaml.FullLoader)

model = model_factory.create_model(cfg)
model.summary()

size = cfg['INPUT']['IMAGE_SIZE']
x = tf.random.normal((1, size, size, 3))
outputs = model(x)
for o in outputs:
    print(o.shape)
