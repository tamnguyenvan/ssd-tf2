"""Create prior boxes for training"""
import sys
from math import sqrt
from itertools import product as product

import tensorflow as tf
import numpy as np


class PriorBox(object):
    def __init__(self, cfg):
        self.feature_maps = cfg['MODEL']['PRIORS']['FEATURE_MAPS']
        self.steps = cfg['MODEL']['PRIORS']['STEPS']
        self.min_sizes = cfg['MODEL']['PRIORS']['MIN_SIZES']
        self.max_sizes = cfg['MODEL']['PRIORS']['MAX_SIZES']
        self.aspect_ratios = cfg['MODEL']['PRIORS']['ASPECT_RATIOS']
        self.clip = cfg['MODEL']['PRIORS']['CLIP']
        self.image_size = cfg['INPUT']['IMAGE_SIZE']

    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f), repeat=2):
                f_k = self.image_size / self.steps[k]

                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                s_k = self.min_sizes[k] / self.image_size
                mean += [cx, cy, s_k, s_k]

                s_k_prime = sqrt(
                    s_k * (self.max_sizes[k] / self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                for ar in self.aspect_ratios[k]:
                    mean += [
                        cx, cy, s_k * sqrt(ar), s_k / sqrt(ar)]
                    mean += [
                        cx, cy, s_k / sqrt(ar), s_k * sqrt(ar)]

        with tf.device('cpu'):
            output = tf.reshape(mean, (-1, 4))
            if self.clip:
                output = tf.clip_by_value(output, 0, 1)
            output = tf.stop_gradient(output)
            return output
