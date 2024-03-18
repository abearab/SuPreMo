#!/usr/bin/env python
# coding: utf-8

import tensorflow.compat.v2 as tf
import tensorflow_hub as hub

enformer_model = hub.load("https://kaggle.com/models/deepmind/enformer/frameworks/TensorFlow2/variations/enformer/versions/1").model

SEQ_LENGTH = 393_216

# Input array [batch_size, SEQ_LENGTH, 4] one hot encoded in order 'ACGT'. The
# `one_hot_encode`. With N values being all zeros.
inputs = tf.zeros((1, SEQ_LENGTH, 4), dtype=tf.float32)
predictions = enformer_model.predict_on_batch(inputs)
predictions['human'].shape  # [batch_size, 896, 5313]
predictions['mouse'].shape  # [batch_size, 896, 1643]

