# Copyright 2022 antillia.com All Rights Reserved.
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


import os
import sys

import numpy as np
import tensorflow as tf


# Based on the following colab
# https://colab.research.google.com/github/google/automl/blob/master/efficientnetv2/tfhub.ipynb#scrollTo=lus9bIA-bQgj
class TestDataset:

  def __init__(self):
    pass

  def create(self, FLAGS):
    data_dir    = FLAGS.data_dir
    image_size  = FLAGS.image_size
    eval_image_size = FLAGS.eval_image_size
    target_size = (image_size, image_size)
    eval_size   = (eval_image_size, eval_image_size)
    #eval_size   = (image_size, image_size)

    #batch_size  = FLAGS.batch_size

    print("---- No data_augumentation ")
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
          rescale            = 1/255,    
       )
    test_generator = test_datagen.flow_from_directory(
             data_dir, 
             target_size   = eval_size, 
             batch_size    = 1,
             interpolation = "bilinear",
             class_mode    = 'categorical',
             shuffle       = False)


    return test_generator

