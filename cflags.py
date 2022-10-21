# Copyright 2021 Google Research. All Rights Reserved.
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
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Command line flags, shared by all executable binaries."""
from absl import flags
FLAGS = flags.FLAGS


def define_flags():
  """A shared function to define flags."""
  flags.DEFINE_string('model_name', 'efficientnetv2-s', 'model name.')
  flags.DEFINE_string('dataset_cfg', 'Imagenet', 'dataset config name.')
  flags.DEFINE_string('hparam_str', '', 'Comma separated k=v pairs of hparams.')
  flags.DEFINE_string('sweeps', '', 'Comma separated k=v pairs for sweeping.')
  flags.DEFINE_bool('use_tpu', True, 'If true, use TPU; otherwise use CPU/GPU.')
  flags.DEFINE_string('tpu_job_name', None, 'job name, default to tpu_worker.')
  # Cloud TPU Cluster Resolvers
  flags.DEFINE_string('tpu', None, 'address e.g. grpc://ip.address.of.tpu:8470')
  flags.DEFINE_string('gcp_project', None, 'Project name.')
  flags.DEFINE_string('tpu_zone', None, 'GCE zone')
  # Model specific flags
  flags.DEFINE_string('data_dir', None, 'The directory for training images.')
  flags.DEFINE_string('eval_name', None, 'Evaluation name.')
  flags.DEFINE_bool('archive_ckpt', True, 'If true, archive the best ckpt.')
  flags.DEFINE_string('model_dir', None, 'Dir for checkpoint and summaries.')
  flags.DEFINE_string('mode', 'train', 'One of {"train", "eval"}.')
  flags.DEFINE_bool('export_to_tpu', False, 'Export metagraph.')

  # 2022/07/20 antillia.com
  flags.DEFINE_bool('channels_first', False, 'Channel first flag.')
  flags.DEFINE_string('ckpt_dir', '', 'Pretrained checkpoint dir.')
  flags.DEFINE_string('data_generator_config', '', 'Image data generator config filepath .')

  flags.DEFINE_string('strategy', 'gpu', 'Strategy: tpu, gpus, gpu.')
  flags.DEFINE_string('best_model_name', 'best_model.5h', 'Best model name.')
  flags.DEFINE_integer('num_classes', 10, 'Number of classes.')
  flags.DEFINE_float('learning_rate',  0.005, 'Learning rate.')
  # 2022/08/14
  flags.DEFINE_float('dropout_rate',  0.3, 'Dropout rate.')

  flags.DEFINE_integer('image_size', 384, 'Image size. 384 if model_name==efficientnetv2-s')
  flags.DEFINE_integer('batch_size', 8, 'Number of batch_size.')
  flags.DEFINE_integer('num_epochs', 10, 'Number of epochs.')
  flags.DEFINE_integer('eval_image_size', None, 'Eval image size.')
  flags.DEFINE_string('eval_dir', './eval', 'Eval directory to save train loss and accuracy.')
  
  flags.DEFINE_bool('data_augmentation', True, 'Data augumentation flag.')
  flags.DEFINE_bool('fine_tuning', True, 'Fine tuning flag.')
  flags.DEFINE_bool('debug', True, 'Debug flag.')
  flags.DEFINE_float('trainable_layers_ratio',  0.3, 'Trainable layers ratio.')
  flags.DEFINE_string('monitor', 'val_loss', 'Early stopping monitor.')
  flags.DEFINE_string('optimizer', 'adam', 'Optimizer.')

  flags.DEFINE_integer('patience', 0, 'Early stopping patience.')
  
  flags.DEFINE_bool('valid_data_augmentation', True, 'Validation data augumentation flag.')
  flags.DEFINE_integer('seed', 123, 'Seed for random.')
