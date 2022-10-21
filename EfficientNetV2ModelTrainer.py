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

# 2022/08/06 Copyright (C) antillia.com
# This is based on github.google/automl/efficientdet/main_tf2.py

# EfficientNetV2ModelTrainer.py

import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_ENABLE_GPU_GARBAGE_COLLECTION"]="false"

import copy
import sys

from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow as tf

import cflags
import datasets

import utils
import hparams
import utils
#import effnetv2_configs
#import effnetv2_model

import shutil
import random as rn

###

# 2022/08/17
#https://github.com/NVIDIA/framework-determinism/blob/master/doc/tensorflow.md
os.environ['PYTHONHASHSEED'] = '0'
#os.environ['TF_DETERMINISTIC_OPS'] = 'true'
os.environ['TF_CUDNN_DETERMINISTIC'] = 'true'

sys.path.append("../../")

from CustomDataset import CustomDataset
from FineTuningModel import FineTuningModel
from EpochChangeCallback import EpochChangeCallback
from ConfusionMatrix import ConfusionMatrix

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
# See also:  #https://tfhub.dev/google/collections/efficientnet_v2/1


class EfficientNetV2ModelTrainer:
  # Constructor
  def __init__(self):
    self.reset_random_seeds(FLAGS.seed)    

    self.model = None
    self.num_epochs= FLAGS.num_epochs # 2 #@param {type:"integer"}
    image_size     = FLAGS.image_size 
    customDataset  = CustomDataset()
    (self.train_generator, self.valid_generator) = customDataset.create(FLAGS)
     
    tf.keras.backend.clear_session()
    class_indices = self.train_generator.class_indices
    #self.class_indices takes the following format:
    # {'cat': 0, 'dog': 1}
    print("--- num_classes {}".format(self.train_generator.num_classes))
    model_name  = FLAGS.model_name
  
    fine_tuning = FLAGS.fine_tuning
    ckpt_dir    = FLAGS.ckpt_dir
    num_classes = len(class_indices)

    trainable_layers_ratio = FLAGS.trainable_layers_ratio
    
    if trainable_layers_ratio < 0.1 or trainable_layers_ratio > 1.0:
       print("--- Set default trainable_layers_ratio=0.3")
       trainable_layers_ratio = 0.3
    print(trainable_layers_ratio)
    finetuning_model = FineTuningModel(model_name, ckpt_dir)
    
    self.model = finetuning_model.build(image_size, 
                                        num_classes,
                                        fine_tuning, 
                                        trainable_layers_ratio = trainable_layers_ratio)
    #self.model.build((None, image_size, image_size, 3))
    if FLAGS.debug:
      self.model.summary()

    model_dir = FLAGS.model_dir
    if not os.path.exists(model_dir):
      os.makedirs(model_dir)
    self.save_train_params(sys.argv, model_dir)

  def reset_random_seeds(self, seed):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    rn.seed(seed)
    tf.compat.v1.set_random_seed(seed)


  def save_train_params(self, argv, dir, section="train"):
    args =  argv[1:]
    filename = section + ".conf"
    filepath = os.path.join(dir, filename)
    NL = "\n"
    with open(filepath, "w") as f:
      f.write("; "+  filename + NL)
      f.write("[" + section + "]"  + NL)
      for arg in args:
        a     = arg.split("=")
        if len(a) == 2:
          key   = a[0]
          value = a[1]
          hypens = "--"
          if key.startswith(hypens):
            key = key[len(hypens):]
          #print(" key {} value {}".format(key, value))
          print("-----------key {} value {}".format(key, value))
          if key == "data_generator_config":
            basename = os.path.basename(value)
            filepath = os.path.join(dir, basename)
            abspath = os.path.abspath(value)
            if os.path.exists(abspath):
              shutil.copy2(abspath, filepath)
            else:
              print("--- Not Found {}".format(abspath))
              raise Exception("Not found " + abspath)
          f.write(key + "=" + str(value) + NL)


  def build_optimizer(self, learning_rate,
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


  def compile(self):
    learning_rate=FLAGS.learning_rate
    # Use adam 
    optimizer = self.build_optimizer(
        learning_rate,  #optimizer_name='adam') 
        optimizer_name=FLAGS.optimizer) # 2022/08/01 'rmsprop')
    
    loss      = tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),
    self.model.compile(
       optimizer = optimizer,
       loss      = loss,
       metrics   = ['accuracy']
      )


  def train_eval(self):
    epch_callback  = EpochChangeCallback(FLAGS.eval_dir)

    model_dir = FLAGS.model_dir
    if not os.path.exists(model_dir):
      os.makedirs(model_dir)                 
    ckpt_path = model_dir + "/best_model.h5" ##'ckpt-{epoch:d}')

    print("----- best_model {}:".format(ckpt_path))
    
    ckpt_callback  = tf.keras.callbacks.ModelCheckpoint(
        ckpt_path,
        verbose           = 1,
        save_best_only    = True,
        save_weights_only = True)
    
    #tb_callback = tf.keras.callbacks.TensorBoard(
    #    log_dir=FLAGS.model_dir, update_freq=100)

    rstr_callback    = utils.ReuableBackupAndRestore(backup_dir=FLAGS.model_dir)

    steps_per_epoch  = self.train_generator.samples // self.train_generator.batch_size
    validation_steps = self.valid_generator.samples // self.valid_generator.batch_size
    callbacks =  [epch_callback, ckpt_callback, rstr_callback]

    # FLAGS.patience default value is 0
    if FLAGS.patience > 0:
      erstp_callback = tf.keras.callbacks.EarlyStopping(monitor  = FLAGS.monitor, 
                                                      patience = FLAGS.patience, 
                                                      verbose  = 2, 
                                                      mode     = 'auto')
      print("--- Add EarlyStopping callback")
      callbacks.append(erstp_callback)

    self.model.fit(
      self.train_generator,
      epochs           = self.num_epochs, 
      steps_per_epoch  = steps_per_epoch,
      validation_data  = self.valid_generator,
      validation_steps = validation_steps,
      callbacks        = callbacks,
      verbose=1
      )

    #2022/08/26
    epch_callback.save_eval_graphs()

    
def main(_) -> None:

  trainer = EfficientNetV2ModelTrainer()
  trainer.compile()
  trainer.train_eval()


if __name__ == '__main__':
  cflags.define_flags()
  app.run(main)
