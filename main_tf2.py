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
"""A simple script to train efficient net with tf2/keras."""

import copy
import os
import re
import sys

from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow as tf

import cflags
import datasets
import effnetv2_configs
import effnetv2_model
import hparams
import utils
sys.path.append("../../")

from CustomDataset import CustomDataset

FLAGS = flags.FLAGS

#2022/07/19
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
        print('{} memory growth: {}'.format(device, tf.config.experimental.get_memory_growth(device)))
else:
    print("Not enough GPU hardware devices available")
#

def build_tf2_optimizer(learning_rate,
                        optimizer_name='rmsprop',
                        decay=0.9,
                        epsilon=0.001,
                        momentum=0.9):
  """Build optimizer."""
  if optimizer_name == 'sgd':
    logging.info('Using SGD optimizer')
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
  elif optimizer_name == 'momentum':
    logging.info('Using Momentum optimizer')
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=learning_rate, momentum=momentum)
  elif optimizer_name == 'rmsprop':
    logging.info('Using RMSProp optimizer')
    optimizer = tf.keras.optimizers.RMSprop(learning_rate, decay, momentum,
                                            epsilon)
  elif optimizer_name == 'adam':
    logging.info('Using Adam optimizer')
    optimizer = tf.keras.optimizers.Adam(learning_rate)
  else:
    raise Exception('Unknown optimizer: %s' % optimizer_name)

  return optimizer


class TrainableModel(effnetv2_model.EffNetV2Model):
  """Wraps efficientnet to make a keras trainable model.

  Handles efficientnet's multiple outputs and adds weight decay.
  """

  def __init__(self,
               model_name='efficientnetv2-s',
               model_config=None,
               name=None,
               weight_decay=0.0):
    super().__init__(
        model_name=model_name,
        model_config=model_config,
        name=name or model_name)

    self.weight_decay = weight_decay

  def _reg_l2_loss(self, weight_decay, regex=r'.*(kernel|weight):0$'):
    """Return regularization l2 loss loss."""
    var_match = re.compile(regex)
    return weight_decay * tf.add_n([
        tf.nn.l2_loss(v)
        for v in self.trainable_variables
        if var_match.match(v.name)
    ])

  def train_step(self, data):
    features, labels = data
    images, labels = features['image'], labels['label']

    with tf.GradientTape() as tape:
      pred = self(images, training=True)
      pred = tf.cast(pred, tf.float32)
      loss = self.compiled_loss(
          labels,
          pred,
          regularization_losses=[self._reg_l2_loss(self.weight_decay)])

    self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
    self.compiled_metrics.update_state(labels, pred)
    return {m.name: m.result() for m in self.metrics}

  def test_step(self, data):
    features, labels = data
    images, labels = features['image'], labels['label']
    pred = self(images, training=False)
    pred = tf.cast(pred, tf.float32)

    self.compiled_loss(
        labels,
        pred,
        regularization_losses=[self._reg_l2_loss(self.weight_decay)])

    self.compiled_metrics.update_state(labels, pred)
    return {m.name: m.result() for m in self.metrics}

# Based on the following colab
# https://colab.research.google.com/github/google/automl/blob/master/efficientnetv2/tfhub.ipynb#scrollTo=lus9bIA-bQgj
"""
class CustomDataset:

  def __init__(self):
    pass

  def create(self, FLAGS):
    data_dir   = FLAGS.data_dir
    image_size = FLAGS.image_size
    batch_size = FLAGS.batch_size
    datagen_kwargs  = dict(rescale=1./255, validation_split=.20)
    dataflow_kwargs = dict(target_size=(image_size, image_size),
                       batch_size=batch_size,
                       interpolation="bilinear")

    valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        **datagen_kwargs)
    valid_generator = valid_datagen.flow_from_directory(
        data_dir, subset="validation", shuffle=False, **dataflow_kwargs)

    do_data_augmentation = True # False #@param {type:"boolean"}
    if do_data_augmentation:
       train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
         rotation_range=40,
         horizontal_flip=True,
         width_shift_range=0.2, height_shift_range=0.2,
         shear_range=0.2, zoom_range=0.2,
         **datagen_kwargs)
       train_generator = train_datagen.flow_from_directory(
        data_dir, subset="training", shuffle=True, **dataflow_kwargs)
      
    else:
      train_datagen = valid_datagen
      train_generator = train_datagen.flow_from_directory(
         data_dir, subset="training", shuffle=True, **dataflow_kwargs)
    return (train_generator, valid_generator)
"""

class EfficientNetV2FineTuningModel:
  def __init__(self, config):
    self.model = None
    self.config = config
    pass

    # whether to finetune the whole model or just the top layer.
    do_fine_tuning = True #@param {type:"boolean"}
    self.num_epochs = FLAGS.num_epochs # 2 #@param {type:"integer"}
    image_size = FLAGS.image_size  #228
    customDataset = CustomDataset()
    (self.train_generator, self.valid_generator) = customDataset.create(FLAGS)

    tf.keras.backend.clear_session()
    """
    model = tf.keras.Sequential([
      # Explicitly define the input shape so the model can be properly
      # loaded by the TFLiteConverter
      tf.keras.layers.InputLayer(input_shape=[image_size, image_size, 3]),
      hub.KerasLayer(hub_url, trainable=do_fine_tuning),
      tf.keras.layers.Dropout(rate=0.2),
      tf.keras.layers.Dense(train_generator.num_classes,
                          kernel_regularizer=tf.keras.regularizers.l2(0.0001))
    ])
    """
    print("--- num_classes {}".format(self.train_generator.num_classes))
    self.model = tf.keras.models.Sequential([
      tf.keras.layers.InputLayer(input_shape=[image_size, image_size, 3]),
      effnetv2_model.get_model('efficientnetv2-b0', include_top=False),
      tf.keras.layers.Dropout(rate=0.2),
      tf.keras.layers.Dense(self.train_generator.num_classes,
                          kernel_regularizer=tf.keras.regularizers.l2(0.0001))
    ])
    self.model.build((None, image_size, image_size, 3))
    self.model.summary()


  def compile(self):
    
    self.model.compile(
      optimizer=tf.keras.optimizers.SGD(learning_rate=0.005, momentum=0.9), 
      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),
      metrics=['accuracy'])
    """
   
    optimizer = build_tf2_optimizer(
        learning_rate, optimizer_name=config.train.optimizer)

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=config.train.label_smoothing, from_logits=True),
        metrics=[
            tf.keras.metrics.TopKCategoricalAccuracy(k=1, name='acc_top1'),
            tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='acc_top5')
        ],
    )
    """


  def train_eval(self):
    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(FLAGS.model_dir, 'ckpt-{epoch:d}'),
        verbose=1,
        save_weights_only=True)
    tb_callback = tf.keras.callbacks.TensorBoard(
        log_dir=FLAGS.model_dir, update_freq=100)
    rstr_callback = utils.ReuableBackupAndRestore(backup_dir=FLAGS.model_dir)

    steps_per_epoch  = self.train_generator.samples // self.train_generator.batch_size
    validation_steps = self.valid_generator.samples // self.valid_generator.batch_size
    # traineval
    hist = self.model.fit(
      self.train_generator,
      epochs=self.num_epochs, steps_per_epoch=steps_per_epoch,
      validation_data=self.valid_generator,
      validation_steps=validation_steps,
      callbacks=[ckpt_callback, tb_callback, rstr_callback],
      verbose=1
      ).history
    print(hist)


def main(_) -> None:
  config = copy.deepcopy(hparams.base_config)
  config.override(effnetv2_configs.get_model_config(FLAGS.model_name))
  config.override(datasets.get_dataset_config(FLAGS.dataset_cfg))
  config.override(FLAGS.hparam_str)
  #config.model.num_classes = config.data.num_classes
  strategy = config.runtime.strategy
  if strategy == 'tpu' and not config.model.bn_type:
    config.model.bn_type = 'tpu_bn'

  # log and save config.
  #logging.info('config=%s', str(config))
  if 'train' in FLAGS.mode:
    if not tf.io.gfile.exists(FLAGS.model_dir):
      tf.io.gfile.makedirs(FLAGS.model_dir)
    config.save_to_yaml(os.path.join(FLAGS.model_dir, 'config.yaml'))

  if strategy == 'tpu':
    tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
    tf.config.experimental_connect_to_cluster(tpu_cluster_resolver)
    tf.tpu.experimental.initialize_tpu_system(tpu_cluster_resolver)
    ds_strategy = tf.distribute.TPUStrategy(tpu_cluster_resolver)
    logging.info('All devices: %s', tf.config.list_logical_devices('TPU'))
  elif strategy == 'gpus':
    ds_strategy = tf.distribute.MirroredStrategy()
    logging.info('All devices: %s', tf.config.list_physical_devices('GPU'))
  else:
    if tf.config.list_physical_devices('GPU'):
      ds_strategy = tf.distribute.MirroredStrategy(['GPU:0'])
    else:
      ds_strategy = tf.distribute.MirroredStrategy(['CPU:0'])

  image_dtype = None
  if config.runtime.mixed_precision:
    print("--------------- mixed_precision is true")
    image_dtype = 'bfloat16' if strategy == 'tpu' else 'float16'
    precision = 'mixed_bfloat16' if strategy == 'tpu' else 'mixed_float16'
    policy = tf.keras.mixed_precision.Policy(precision)
    tf.keras.mixed_precision.set_global_policy(policy)

  finetuning_model = EfficientNetV2FineTuningModel(config)
  finetuning_model.compile()
  finetuning_model.train_eval()


def __main(_) -> None:
  config = copy.deepcopy(hparams.base_config)
  config.override(effnetv2_configs.get_model_config(FLAGS.model_name))
  config.override(datasets.get_dataset_config(FLAGS.dataset_cfg))
  config.override(FLAGS.hparam_str)
  config.model.num_classes = config.data.num_classes
  strategy = config.runtime.strategy
  if strategy == 'tpu' and not config.model.bn_type:
    config.model.bn_type = 'tpu_bn'

  # log and save config.
  
  logging.info('config=%s', str(config))
  if 'train' in FLAGS.mode:
    if not tf.io.gfile.exists(FLAGS.model_dir):
      tf.io.gfile.makedirs(FLAGS.model_dir)
    config.save_to_yaml(os.path.join(FLAGS.model_dir, 'config.yaml'))

  if strategy == 'tpu':
    tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
    tf.config.experimental_connect_to_cluster(tpu_cluster_resolver)
    tf.tpu.experimental.initialize_tpu_system(tpu_cluster_resolver)
    ds_strategy = tf.distribute.TPUStrategy(tpu_cluster_resolver)
    logging.info('All devices: %s', tf.config.list_logical_devices('TPU'))
  elif strategy == 'gpus':
    ds_strategy = tf.distribute.MirroredStrategy()
    logging.info('All devices: %s', tf.config.list_physical_devices('GPU'))
  else:
    if tf.config.list_physical_devices('GPU'):
      ds_strategy = tf.distribute.MirroredStrategy(['GPU:0'])
    else:
      ds_strategy = tf.distribute.MirroredStrategy(['CPU:0'])

  with ds_strategy.scope():
    train_split = config.train.split or 'train'
    eval_split = config.eval.split or 'eval'
    num_train_images = config.data.splits[train_split].num_images
    num_eval_images = config.data.splits[eval_split].num_images

    train_size = config.train.isize
    eval_size = config.eval.isize
    if train_size <= 16.:
      train_size = int(eval_size * train_size) // 16 * 16

    image_dtype = None
    if config.runtime.mixed_precision:
      print("--------------- mixed_precision is true")
      image_dtype = 'bfloat16' if strategy == 'tpu' else 'float16'
      precision = 'mixed_bfloat16' if strategy == 'tpu' else 'mixed_float16'
      policy = tf.keras.mixed_precision.Policy(precision)
      tf.keras.mixed_precision.set_global_policy(policy)

    model = TrainableModel(
        config.model.model_name,
        config.model,
        weight_decay=config.train.weight_decay)

    if config.train.ft_init_ckpt:  # load pretrained ckpt for finetuning.
      model(tf.keras.Input([None, None, 3]))
      ckpt = config.train.ft_init_ckpt
      utils.restore_tf2_ckpt(model, ckpt, exclude_layers=('_fc', 'optimizer'))

    steps_per_epoch = num_train_images // config.train.batch_size
    total_steps = steps_per_epoch * config.train.epochs

    scaled_lr = config.train.lr_base * (config.train.batch_size / 256.0)
    scaled_lr_min = config.train.lr_min * (config.train.batch_size / 256.0)
    learning_rate = utils.WarmupLearningRateSchedule(
        scaled_lr,
        steps_per_epoch=steps_per_epoch,
        decay_epochs=config.train.lr_decay_epoch,
        warmup_epochs=config.train.lr_warmup_epoch,
        decay_factor=config.train.lr_decay_factor,
        lr_decay_type=config.train.lr_sched,
        total_steps=total_steps,
        minimal_lr=scaled_lr_min)

    optimizer = build_tf2_optimizer(
        learning_rate, optimizer_name=config.train.optimizer)

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=config.train.label_smoothing, from_logits=True),
        metrics=[
            tf.keras.metrics.TopKCategoricalAccuracy(k=1, name='acc_top1'),
            tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='acc_top5')
        ],
    )

    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(FLAGS.model_dir, 'ckpt-{epoch:d}'),
        verbose=1,
        save_weights_only=True)
    tb_callback = tf.keras.callbacks.TensorBoard(
        log_dir=FLAGS.model_dir, update_freq=100)
    rstr_callback = utils.ReuableBackupAndRestore(backup_dir=FLAGS.model_dir)

    def filter_callbacks(callbacks):
      if strategy == 'tpu' and not FLAGS.model_dir.startswith('gs://'):
        return list(filter(lambda callback: isinstance(callback, tf.keras.callbacks.ModelCheckpoint), callbacks))
      return callbacks

    def get_dataset(training, image_size, config):
      """A shared utility to get input dataset."""
      if training:
        return ds_strategy.distribute_datasets_from_function(
            datasets.build_dataset_input(
                True, image_size, image_dtype, FLAGS.data_dir, train_split,
                config.data).distribute_dataset_fn(config.train.batch_size))
      else:
        return ds_strategy.distribute_datasets_from_function(
            datasets.build_dataset_input(
                False, image_size, image_dtype, FLAGS.data_dir, eval_split,
                config.data).distribute_dataset_fn(config.eval.batch_size))

    if FLAGS.mode == 'traineval':
      model.fit(
          get_dataset(training=True, image_size=train_size, config=config),
          epochs=config.train.epochs,
          steps_per_epoch=steps_per_epoch,
          validation_data=get_dataset(
              training=False, image_size=eval_size, config=config),
          validation_steps=num_eval_images // config.eval.batch_size,
          callbacks=filter_callbacks([ckpt_callback, tb_callback, rstr_callback]),
          # don't log spam if running on tpus
          verbose=2 if strategy == 'tpu' else 1,
      )
    elif FLAGS.mode == 'train':
      if not config.train.stages:
        model.fit(
            get_dataset(training=True, image_size=train_size, config=config),
            epochs=config.train.epochs,
            steps_per_epoch=steps_per_epoch,
            callbacks=filter_callbacks([ckpt_callback, tb_callback, rstr_callback]),
            verbose=2 if strategy == 'tpu' else 1,
        )
      else:
        total_stages = config.train.stages
        ibase = config.data.ibase or (train_size / 2)

        if config.train.sched:
          ram_list = np.linspace(5, config.data.ram, total_stages)
          mixup_list = np.linspace(0, config.data.mixup_alpha, total_stages)
          cutmix_list = np.linspace(0, config.data.cutmix_alpha, total_stages)

        for stage in range(total_stages):
          ratio = float(stage + 1) / float(total_stages)
          start_epoch = int(
              float(stage) / float(total_stages) * config.train.epochs)
          end_epoch = int(ratio * config.train.epochs)
          image_size = int(ibase + (train_size - ibase) * ratio)

          if config.train.sched:
            config.data.ram = ram_list[stage]
            config.data.mixup_alpha = mixup_list[stage]
            config.data.cutmix_alpha = cutmix_list[stage]

          model.fit(
              get_dataset(training=True, image_size=image_size, config=config),
              initial_epoch=start_epoch,
              epochs=end_epoch,
              steps_per_epoch=steps_per_epoch,
              callbacks=filter_callbacks([ckpt_callback, tb_callback, rstr_callback]),
              verbose=2 if strategy == 'tpu' else 1,
          )
    elif FLAGS.mode == 'eval':
      for ckpt in tf.train.checkpoints_iterator(
          FLAGS.model_dir, timeout=60 * 60 * 24):
        model.load_weights(ckpt)
        eval_results = model.evaluate(
            get_dataset(training=False, image_size=eval_size, config=config),
            batch_size=config.eval.batch_size,
            steps=num_eval_images // config.eval.batch_size,
            callbacks=filter_callbacks([tb_callback, rstr_callback]),
            verbose=2 if strategy == 'tpu' else 1,
        )

        try:
          current_epoch = int(os.path.basename(ckpt).split('-')[1])
        except IndexError:
          logging.info('%s has no epoch info: stop!', ckpt)
          break

        logging.info('Epoch: %d, total %d', current_epoch, config.train.epochs)
        if current_epoch >= config.train.epochs:
          break
    else:
      raise ValueError(f'Invalid mode {FLAGS.mode}')


if __name__ == '__main__':
  cflags.define_flags()
  app.run(main)
