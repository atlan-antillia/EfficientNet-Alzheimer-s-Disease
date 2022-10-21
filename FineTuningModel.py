# Copyright 2022 (C) antillia.com. All Rights Reserved.
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

# 2022/07/20 Copyright (C) antillia.com

# FineTuningModel.py

#import copy
import os
#import re
import sys

from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow as tf


import numpy as np
import random as rn
from effnetv2_model import EffNetV2Model
FLAGS = flags.FLAGS

import cflags
import datasets
import effnetv2_configs
import effnetv2_model
import hparams
import utils

sys.path.append("../../")


#Please specify --fine_tuning=False, 
# If your would like to create a transfer-learning model,
# please specify --fine_tuning=False 

class FineTuningModel:
  #Constructor
  def __init__(self, model_name, pretrained_ckpt, debug=True):
 
    self.model = None
    self.debug = debug
    #self.channels_first = channels_first

    #self.base_model = effnetv2_model.get_model(model_name, include_top=False)
    
    self.base_model = EffNetV2Model(model_name, include_top=False)
    
    input_shape = (None, None, 3)

    self.base_model(tf.keras.Input(shape=input_shape),
      training       = True,
      with_endpoints = False)

    if pretrained_ckpt !=None:
      if tf.io.gfile.isdir(pretrained_ckpt):
        pretrained_ckpt = tf.train.latest_checkpoint(pretrained_ckpt)
      self.base_model.load_weights(pretrained_ckpt)
      print("--- loaded weight {}".format(pretrained_ckpt))
      

  def show_base_model_layers(self):
    print("--- EfficientNetV2")
    if self.debug:
      for i, layer in enumerate(self.base_model.layers):
        print("---  i: {} name: {} trainable:{}".format(i, layer.name, layer.trainable))


  def show_customized_model_layers(self):
    print("--- Customized EfficientNetV2")
    if self.debug:
      for i, layer in enumerate(self.model.layers):
        print("---  i: {} name: {} trainable:{}".format(i, layer.name, layer.trainable))

  def build(self, image_size, num_classes, fine_tuning, trainable_layers_ratio=0.3):

    num_layers = len(self.base_model.layers)
    print("--- num_layers {}".format(num_layers))

    if fine_tuning:
      non_trainable_layers_ratio = 1.0 - trainable_layers_ratio
      non_trainable_max_layers = int( float(num_layers) * non_trainable_layers_ratio)
      print("--- non_trainable_max_layers {}".format(non_trainable_max_layers))     

      self.base_model.trainable = True

      for i, layer in enumerate(self.base_model.layers):
        if (i < non_trainable_max_layers):
          layer.trainable = False
        else:
          layer.trainable = True
    else:
      # 2022/08/16 Transfer Learning
      self.base_model.trainable = False
    
    self.show_base_model_layers()
    input_shape = [image_size, image_size, 3]

    self.model = tf.keras.models.Sequential([
      tf.keras.layers.InputLayer(input_shape=input_shape, name='image', dtype=tf.float32),
      self.base_model,
      tf.keras.layers.Dropout(FLAGS.dropout_rate),
      #tf.keras.layers.Dropout(rate=0.3),

      tf.keras.layers.Dense(num_classes, 
                          name       = "predictions",
                          kernel_regularizer=tf.keras.regularizers.l2(0.0001)
                          )
                          
    ])
    
    self.show_customized_model_layers()

    return self.model

