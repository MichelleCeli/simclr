# coding=utf-8
# Copyright 2020 The SimCLR Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific simclr governing permissions and
# limitations under the License.
# ==============================================================================
"""Data pipeline."""

import functools
from absl import flags
from absl import logging

import data_util
import tensorflow.compat.v2 as tf

from dataset_loader import create_dataset

FLAGS = flags.FLAGS


def build_input_fn(data_dir, global_batch_size, topology, is_training,
                   csv_path=None):
  """Build input function.

  Args:
    data_dir: directory for specified dataset.
    global_batch_size: Global batch size.
    topology: An instance of `tf.tpu.experimental.Topology` or None.
    is_training: Whether to build in training mode.
    csv_path: Eigene Änderung. Optionaler expliziter Pfad zur Label-CSV.
      Falls None (Standard), wird wie bisher FLAGS.csv_path verwendet -> das
      Trainingsverhalten ändert sich dadurch nicht. Wird gebraucht, damit die
      Evaluation eine ANDERE CSV (FLAGS.test_csv_path) laden kann als das
      Training, ohne dass beide sich eine globale FLAGS.csv_path teilen
      müssen.

  Returns:
    A function that accepts a dict of params and returns a tuple of images and
    features, to be used as the input_fn in TPUEstimator.
  """

  def _input_fn(input_context):
    """Inner input function."""
    batch_size = input_context.get_per_replica_batch_size(global_batch_size)
    logging.info('Global batch size: %d', global_batch_size)
    logging.info('Per-replica batch size: %d', batch_size)
    preprocess_fn_pretrain = get_preprocess_fn(is_training, is_pretrain=True)
    preprocess_fn_finetune = get_preprocess_fn(is_training, is_pretrain=False)
    # num_classes = builder.info.features['label'].num_classes

    # Labels from tf datasets used in SimCLR
    '''
    def map_fn(image, label):
      """Produces multiple transformations of the same batch."""
      if is_training and FLAGS.train_mode == 'pretrain':
        xs = []
        for _ in range(2):  # Two transformations
          xs.append(preprocess_fn_pretrain(image))
        image = tf.concat(xs, -1)
      else:
        image = preprocess_fn_finetune(image)
      label = tf.one_hot(label, num_classes)
      return image, label
    '''

    def map_fn(image, label):

        if is_training and FLAGS.train_mode == 'pretrain':
            xs = []
            for _ in range(2):
                xs.append(preprocess_fn_pretrain(image))
            image = tf.concat(xs, -1)
        else:
            image = preprocess_fn_finetune(image)
        
        return image, label

    # SimCLR datasetbuilder with tensorflow datasets
    # logging.info('num_input_pipelines: %d', input_context.num_input_pipelines)
    '''
    dataset = builder.as_dataset(
        split=FLAGS.train_split if is_training else FLAGS.eval_split,
        shuffle_files=is_training,
        as_supervised=True,
        # Passing the input_context to TFDS makes TFDS read different parts
        # of the dataset on different workers. We also adjust the interleave
        # parameters to achieve better performance.
        read_config=tfds.ReadConfig(
            interleave_cycle_length=32,
            interleave_block_length=1,
            input_context=input_context))
    if FLAGS.cache_dataset:
      dataset = dataset.cache()
    '''

    # Eigene Änderung: csv_path kommt jetzt primär aus dem Funktionsparameter,
    # Fallback auf FLAGS.csv_path nur wenn keiner übergeben wurde (siehe
    # Docstring oben). Vorher stand hier direkt FLAGS.csv_path, wodurch
    # Training und Evaluation zwangsläufig dieselbe CSV benutzt hätten.
    resolved_csv_path = csv_path if csv_path is not None else FLAGS.csv_path
    dataset = create_dataset(data_dir=data_dir, image_size=FLAGS.image_size, csv_path=resolved_csv_path)

    if is_training:
      options = tf.data.Options()
      options.experimental_deterministic = False
      options.experimental_slack = True
      dataset = dataset.with_options(options)
      buffer_multiplier = 50 if FLAGS.image_size <= 32 else 10
      dataset = dataset.shuffle(batch_size * buffer_multiplier)
      dataset = dataset.repeat(-1)
    dataset = dataset.map(
        map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=is_training)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

  return _input_fn


def build_distributed_dataset(data_dir, batch_size, is_training, strategy,
                              topology, csv_path=None):

    # SimCLR
    '''
    input_fn = build_input_fn(builder, batch_size, topology, is_training)
    '''

    # Eigene Änderung: csv_path wird optional durchgereicht (siehe
    # build_input_fn oben). Default None -> Verhalten für bestehende
    # Aufrufer (Training) bleibt unverändert (FLAGS.csv_path).
    input_fn = build_input_fn(
        data_dir,
        batch_size,
        topology,
        is_training,
        csv_path
    )

    return strategy.distribute_datasets_from_function(input_fn)


def get_preprocess_fn(is_training, is_pretrain):
  """Get function that accepts an image and returns a preprocessed image."""
  # Disable test cropping for small images (e.g. CIFAR)
  if FLAGS.image_size <= 32:
    test_crop = False
  else:
    test_crop = True
  color_jitter_strength = FLAGS.color_jitter_strength if is_pretrain else 0.
  return functools.partial(
      data_util.preprocess_image,
      height=FLAGS.image_size,
      width=FLAGS.image_size,
      is_training=is_training,
      color_jitter_strength=color_jitter_strength,
      test_crop=test_crop)
