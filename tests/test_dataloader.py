import time
import cv2
import numpy as np

import context
import tensorflow as tf
from dataset.coco import DataLoader
from dataset.prior_boxes import PriorBox
from utils import box_utils, coco_utils
from config import cfg


tf.debugging.enable_check_numerics(True)

COLOR_MAP = np.random.randint(0, 256, (1000, 3))

prior_boxes = PriorBox().forward()
batch_size = 32
loader = DataLoader(prior_boxes, 32)

# tfrecord_file = 'data/coco2017_train.tfrecord'
image_dir = 'data/train2017'
anno_path = 'data/annotations/instances_train2017.json'
label_map_file = 'data/coco.names'
dataset = loader.load(image_dir, label_map_file, anno_path)

start = time.time()
for i, batch in enumerate(dataset):
    print('Batch', i)
print('Took {:.2f}s'.format(time.time() - start))

# for batch in dataset.take(1):
#     pass
#
# images = batch[0]
# image = images[0].numpy()
# image = (image * 127.5) + 127.5
# image = np.clip(image, 0, 255)
# image = image.astype('uint8')
# image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
# h, w = image.shape[:2]
#
#
# labels = batch[1][0].numpy()
# bboxes = batch[2][0]
# print(bboxes)
# bboxes = box_utils.decode(prior_boxes, bboxes)
# bboxes = bboxes.numpy()
# label_map = coco_utils.load_label_map('./data/coco.names')
# for bbox, label in zip(bboxes, labels):
#     bbox = list(map(lambda x: int(x * h), bbox))
#     if label > 0:
#         COLOR = tuple([int(x) for x in COLOR_MAP[np.random.choice(len(COLOR_MAP))]])
#         cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), COLOR, 2)
#         cv2.putText(image, str(label_map[label]), (bbox[0], bbox[1] - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR, 3)
#
# cv2.imshow('img', image)
# cv2.waitKey(0)
