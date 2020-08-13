"""Module for box utils"""
import tensorflow as tf


def to_point_form(box):
    """Convert center-form box to point-form"""
    return tf.concat([
        box[..., :2] - box[..., 2:4] / 2,
        box[..., :2] + box[..., 2:4] / 2
    ], axis=-1)


def to_center_form(box):
    """Convert point-form to center-form"""
    return tf.concat([
        (box[..., :2] + box[..., 2:]) / 2,
        box[..., 2:] - box[..., :2]
    ], axis=-1)


def compute_area(top_left, bottom_right):
    """Compute box area based on corners"""
    wh = tf.clip_by_value(bottom_right - top_left, 0., 512.)
    area = wh[..., 0] * wh[..., 1]
    return area


def broadcast_iou(box1, box2):
    """Calculate overlaping between two boxes broadcastly

    Args:
      :box1: Prior boxes coordinates, shape of (num_priors, 4)
      :box2: Ground truth boxes coodinates, shape of (max_boxes, 4)
    """
    box1 = tf.expand_dims(box1, axis=1)
    box2 = tf.expand_dims(box2, axis=0)

    box1_area = compute_area(box1[..., :2], box1[..., 2:])
    box2_area = compute_area(box2[..., :2], box2[..., 2:])
    top_left = tf.maximum(box1[..., :2], box2[..., :2])
    bottom_right = tf.minimum(box1[..., 2:], box2[..., 2:])
    overlap_area = compute_area(top_left, bottom_right)
    iou = overlap_area / (box1_area + box2_area - overlap_area)
    return iou


def encode(priors, boxes, variances=[0.1, 0.2]):
    """
    """
    transformed_boxes = to_center_form(boxes)
    return tf.concat([
        (transformed_boxes[..., :2] - priors[..., :2])
        / (priors[..., 2:] * variances[0]),
        tf.math.log(transformed_boxes[..., 2:] / priors[..., 2:]) / variances[1]
    ], axis=-1)


def decode(priors, locs, variances=[0.1, 0.2]):
    """
    """
    locs = tf.concat([
        locs[..., :2] * priors[..., 2:] * variances[0] + priors[..., :2],
        tf.math.exp(locs[..., 2:] * variances[1]) * priors[..., 2:]
    ], axis=-1)
    boxes = to_point_form(locs)
    return boxes


def compute_targets(priors, gt_boxes, gt_labels, iou_threshold=0.5):
    """
    """
    iou = broadcast_iou(to_point_form(priors), gt_boxes)

    best_gt_iou = tf.reduce_max(iou, 1)
    best_gt_idx = tf.argmax(iou, 1)

    # best_prior_iou = tf.reduce_max(iou, 0)
    best_prior_idx = tf.argmax(iou, 0)

    best_gt_idx = tf.tensor_scatter_nd_update(
        best_gt_idx,
        tf.expand_dims(best_prior_idx, 1),
        tf.range(tf.shape(best_prior_idx)[0], dtype=tf.int64))

    best_gt_iou = tf.tensor_scatter_nd_update(
        best_gt_iou,
        tf.expand_dims(best_prior_idx, 1),
        tf.ones_like(best_prior_idx, dtype=tf.float32))

    gt_confs = tf.gather(gt_labels, best_gt_idx)
    gt_confs = tf.where(
        tf.less(best_gt_iou, iou_threshold),
        tf.zeros_like(gt_confs),
        gt_confs)

    gt_boxes = tf.gather(gt_boxes, best_gt_idx)
    gt_locs = encode(priors, gt_boxes)

    return gt_confs, gt_locs


def compute_nms(boxes, scores, nms_threshold, limit=200):
    """ Perform Non Maximum Suppression algorithm
        to eliminate boxes with high overlap
    Args:
        boxes: tensor (num_boxes, 4)
               of format (xmin, ymin, xmax, ymax)
        scores: tensor (num_boxes,)
        nms_threshold: NMS threshold
        limit: maximum number of boxes to keep
    Returns:
        idx: indices of kept boxes
    """
    if boxes.shape[0] == 0:
        return tf.constant([], dtype=tf.int32)
    selected = [0]
    idx = tf.argsort(scores, direction='DESCENDING')
    idx = idx[:limit]
    boxes = tf.gather(boxes, idx)

    iou = broadcast_iou(boxes, boxes)

    while True:
        row = iou[selected[-1]]
        next_indices = row <= nms_threshold
        # iou[:, ~next_indices] = 1.0
        iou = tf.where(
            tf.expand_dims(tf.math.logical_not(next_indices), 0),
            tf.ones_like(iou, dtype=tf.float32),
            iou)

        if not tf.math.reduce_any(next_indices):
            break

        selected.append(tf.argsort(
            tf.dtypes.cast(next_indices, tf.int32), direction='DESCENDING')[0].numpy())

    return tf.gather(idx, selected)
