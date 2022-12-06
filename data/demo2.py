import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import h5py
from tqdm import tqdm
# import torch
# import tensorflow as tf
# tf.compat.v1.disable_eager_execution()
# tfrecord_path = "clevr_with_masks_train.tfrecords"
# raw_dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type = 'GZIP')
# raw_dataset = raw_dataset.take(10)
# batched_dataset = raw_dataset.batch(4)
# iterator = tf.compat.v1.data.make_one_shot_iterator(batched_dataset)
# data = iterator.get_next()
# with tf.compat.v1.Session() as sess:
#     batch = sess.run(data)
#     print(batch)
#     print(batch.shape)
#     image_batch = batch.pop('image')
#     print(image_batch)
#     print(image_batch.shape)

# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""CLEVR (with masks) dataset reader."""

import tensorflow.compat.v1 as tf


COMPRESSION_TYPE = tf.io.TFRecordOptions.get_compression_type_string('GZIP')
IMAGE_SIZE = [64, 64]
# The maximum number of foreground and background entities in the provided
# dataset. This corresponds to the number of segmentation masks returned per
# scene.

BYTE_FEATURES = ['mask', 'image']

# Create a dictionary mapping feature names to `tf.Example`-compatible
# shape and data type descriptors.
features = {
    'image': tf.FixedLenFeature(IMAGE_SIZE+[3], tf.string),
    'mask': tf.FixedLenFeature([7]+IMAGE_SIZE+[1], tf.string),
}


def _decode(example_proto):
    # Parse the input `tf.Example` proto using the feature description dict above.
    single_example = tf.parse_single_example(example_proto, features)
    for k in BYTE_FEATURES:
        single_example[k] = tf.squeeze(tf.decode_raw(single_example[k], tf.uint8),axis=-1)
    return single_example


def dataset(tfrecords_path, read_buffer_size=None, map_parallel_calls=None):
    """Read, decompress, and parse the TFRecords file.
    Args:
    tfrecords_path: str. Path to the dataset file.
    read_buffer_size: int. Number of bytes in the read buffer. See documentation
        for `tf.data.TFRecordDataset.__init__`.
    map_parallel_calls: int. Number of elements decoded asynchronously in
        parallel. See documentation for `tf.data.Dataset.map`.
    Returns:
    An unbatched `tf.data.TFRecordDataset`.
    """
    raw_dataset = tf.data.TFRecordDataset(
        tfrecords_path, compression_type=COMPRESSION_TYPE,
        buffer_size=read_buffer_size)
    return raw_dataset.map(_decode, num_parallel_calls=map_parallel_calls)

d = dataset("objects_room_train.tfrecords")

for i in tqdm(range(0,200)):
    
    images = np.zeros((5000, 3, 64, 64))
    masks = np.zeros((5000, 7, 64, 64))

    for j,raw_record in enumerate(d.take(5000)):
    
        image = raw_record['image'].numpy()
        images[j,...] = np.moveaxis(image,-1,0)

        mask_orig = raw_record['mask'].numpy()
        mask = np.zeros((mask_orig.shape[0], 64, 64))
        for k,mask_i in enumerate(mask_orig):
            mask[k,...] = mask_i[:,:,0]
        masks[j,...] = mask

    hf = h5py.File(f'object-room/{i}.h5', 'w')
    hf.create_dataset('images', data=images)
    hf.create_dataset('masks', data=masks)
    hf.close()

    hf = h5py.File(f'object-room/{i}.h5', 'r')
    print(hf.keys())
    print(hf['images'].shape)
    print(hf['masks'].shape)