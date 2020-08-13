"""
"""
import os
import json
import numpy as np
import tensorflow as tf

from utils import coco_utils, box_utils


class DataLoader:
    """Abstract class for COCO data loader"""
    def __init__(self, prior_boxes, batch_size=32, num_workers=None,
                 image_size=None, max_boxes=100, training=True):
        """
        Args
          :prior_boxes: The default boxes.
          :batch_size: The batch size.
          :num_workers: Number of parallel workers.
          :image_size: The input image size.
          :max_boxes: Maxium boxes per image.
          :training: Training phase.
        """
        self.prior_boxes = prior_boxes
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.max_boxes = max_boxes
        self.training = training

    def _get_anno(self, image_dir, label_map_file, annotations_file):
        """Load annotations from image directory and annotation file.

        Args
          :image_dir: The image directory consists of train2017 and val2017.
          :label_map_file: Path to COCO label map file.
          :annotations_file: Path to annotations file.

        Returns

        """
        train_text = 'train' if self.training else 'val'
        cache_file = f'/tmp/coco_{train_text}.npz'
        if os.path.exists(cache_file):
            data = np.load(cache_file)
            filenames = data.get('filenames')
            bboxes = data.get('bboxes')
            labels = data.get('labels')
            if (filenames is None or bboxes is None or labels is None):
                print('Cache data corrupted, reload from raw data')
            return list(filenames), list(bboxes), list(labels)

        annotations_json = json.load(open(annotations_file))
        images = annotations_json['images']
        annotations = annotations_json['annotations']
        categories = annotations_json['categories']

        # Create class_id -> class_name map
        if not os.path.exists(label_map_file):
            raise FileNotFoundError(
                'Not found COCO label file: {label_map_file}')

        label_map = coco_utils.load_label_map(label_map_file)
        id_map = {name: idx for idx, name in label_map.items()}
        category_meta = {}
        for category_data in categories:
            raw_class_id = category_data['id']
            class_name = category_data['name']
            class_id = id_map[class_name]
            category_meta[raw_class_id] = {
                'class_name': class_name,
                'class_id': class_id
            }

        # Create image_id -> (filename, width, height) map
        image_meta = {}
        for image_data in images:
            image_id = image_data['id']
            filename = image_data['file_name']
            width = image_data['width']
            height = image_data['height']
            image_meta[image_id] = {
                'filename': filename,
                'width': width,
                'height': height
            }

        # Create image_id -> annotation data map
        data = {}
        for annotation_data in annotations:
            image_id = annotation_data['image_id']
            filename = image_meta[image_id]['filename']
            width = image_meta[image_id]['width']
            height = image_meta[image_id]['height']
            if width <= 0 or height <= 0:
                # print(f'Invalid image width or height: {width}'
                #       f' {height} {filename}')
                continue

            # Check valid box
            bbox = list(map(int, annotation_data['bbox']))
            if bbox[2] <= 0 or bbox[3] <= 0:
                # print(f'Invalid box width: {bbox[2]} height: {bbox[3]}'
                #       f' image: {filename}')
                continue

            # Normalize it
            bbox = [bbox[0] / width, bbox[1] / height,
                    (bbox[0] + bbox[2]) / width,
                    (bbox[1] + bbox[3]) / height]

            raw_class_id = annotation_data['category_id']
            class_name = category_meta[raw_class_id]['class_name']
            class_id = category_meta[raw_class_id]['class_id']
            if image_id not in data:
                data[image_id] = {
                    'filename': os.path.join(image_dir, filename),
                    'bboxes': [],
                    'classes': [],
                    'names': [],
                    'width': width,
                    'height': height
                }
            data[image_id]['bboxes'].append(bbox)
            data[image_id]['classes'].append(class_id)
            data[image_id]['names'].append(class_name)

        filenames = []
        bboxes = []
        labels = []
        for image_id, image_data in data.items():
            filenames.append(image_data['filename'])

            boxes = image_data['bboxes']
            classes = image_data['classes']
            padded_len = max(0, self.max_boxes - len(boxes))
            fixed_boxes = boxes + [[0., 0., 0., 0.]] * padded_len
            bboxes.append(fixed_boxes[:self.max_boxes])
            fixed_labels = classes + [0] * padded_len
            labels.append(fixed_labels[:self.max_boxes])

        with open(cache_file, 'wb') as f:
            np.savez(f, filenames=filenames, bboxes=bboxes, labels=labels)
        return filenames, bboxes, labels

    def _parse_fn(self, img_path, bboxes, labels):
        """Parse image path, boxes and labels to targets.

        Args
          :img_path: Path to the image.
          :bboxes: Tensor of shape (max_boxes, 4).
          :labels: Tensor of shape (max_boxes,).
        """
        image = tf.image.decode_jpeg(tf.io.read_file(img_path, 'rb'), 3)
        image = tf.image.resize(image, (self.image_size, self.image_size))
        bboxes = tf.cast(bboxes, tf.float32)
        labels = tf.cast(labels, tf.int64)

        # Truncated the zero-padding
        bboxes = bboxes[labels > 0]
        labels = labels[labels > 0]

        gt_confs, gt_locs = box_utils.compute_targets(
            self.prior_boxes, bboxes, labels)
        return image, gt_confs, gt_locs

    def _transform_images(self, images, gt_confs, gt_locs):
        """Transform the input images."""
        images = (images - 127.5) / 127.5
        return images, gt_confs, gt_locs

    def load(self, image_dir, label_map_file, anno_path):
        """Load data and create tf dataset.

        Args
          :image_dir: Image directory consists of train2017 and val2017.
          :label_map_file: Path to COCO label map file.
          :annotations_file: Path to annotations file.

        Returns
          :dataset: A tf dataset object.
          :length: Length of the dataset.
        """
        # Load annotations
        filenames, bboxes, labels = self._get_anno(
            image_dir, label_map_file, anno_path)
        dataset = tf.data.Dataset.from_tensor_slices(
            (filenames, bboxes, labels))
        if self.training:
            dataset = dataset.shuffle(1000)

        AUTO = tf.data.experimental.AUTOTUNE
        if not isinstance(self.num_workers, int):
            self.num_workers = AUTO
        dataset = dataset.map(
            self._parse_fn, num_parallel_calls=self.num_workers)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.map(
            self._transform_images, num_parallel_calls=self.num_workers)
        dataset = dataset.prefetch(AUTO)
        return dataset, len(filenames)
