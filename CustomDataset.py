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
sys.path.append("../../")

from ConfigParser import ConfigParser

import numpy as np
import tensorflow as tf


# Based on the following colab
# https://colab.research.google.com/github/google/automl/blob/master/efficientnetv2/tfhub.ipynb#scrollTo=lus9bIA-bQgj
class CustomDataset:

  def __init__(self):
    pass


  def create(self, FLAGS):
    seed        = FLAGS.seed
    data_dir    = FLAGS.data_dir
    image_size  = FLAGS.image_size
    #eval_image_size = FLAGS.eval_image_size
    target_size = (image_size, image_size)
    #eval_size   = (eval_image_size, eval_image_size)
    eval_size   = (image_size, image_size)
    data_generator_config = FLAGS.data_generator_config
    parser      = ConfigParser(data_generator_config)

    batch_size  = FLAGS.batch_size

    TRAINING   = "training"
    VALIDATION = "validation"

    data_augmentation = FLAGS.data_augmentation # 
    if data_augmentation:
       
       print("---- Do data_augumentation")
       train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
          rescale            = 1/255,  
          validation_split   = parser.get(TRAINING, "validation_split", 0.2),

          featurewise_center   = parser.get(TRAINING, "featurewise_center", False),
          samplewise_center    = parser.get(TRAINING, "samplewise_center",  False),
          featurewise_std_normalization = parser.get(TRAINING, "featurewise_std_normalization", False),
          samplewise_std_normalization  = parser.get(TRAINING, "samplewise_std_normalization", False),
          zca_whitening                 = parser.get(TRAINING, "zca_whitening",                False),
   
          rotation_range     = parser.get(TRAINING, "rotation_range", 8),
          horizontal_flip    = parser.get(TRAINING, "horizontal_flip", True),
          vertical_flip      = parser.get(TRAINING, "vertical_flip", True),

          width_shift_range  = parser.get(TRAINING, "width_shift_range", 0.9), 
          height_shift_range = parser.get(TRAINING, "height_shift_range", 0.9),
          shear_range        = parser.get(TRAINING, "shear_range", 0.1), 
          zoom_range         = parser.get(TRAINING, "zoom_range", 0.1), 
          #brightness_range   = parser.get(TRAINING, "brightness_range", None)
          )
       train_generator = train_datagen.flow_from_directory(
             data_dir, 
             target_size   = target_size, 
             batch_size    = batch_size,
             interpolation = "bilinear",
             subset        = "training",
             seed          = seed, 
             shuffle       = True)
       if FLAGS.valid_data_augmentation == True:
         valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale            = 1/255,  
            validation_split   = parser.get(VALIDATION, "validation_split", 0.2),

            featurewise_center   = parser.get(VALIDATION, "featurewise_center", False),
            samplewise_center    = parser.get(VALIDATION, "samplewise_center",  False),
            featurewise_std_normalization = parser.get(VALIDATION, "featurewise_std_normalization", False),
            samplewise_std_normalization  = parser.get(VALIDATION, "samplewise_std_normalization", False),
            zca_whitening                 = parser.get(VALIDATION, "zca_whitening",                False),

            rotation_range     = parser.get(VALIDATION, "rotation_range", 8),
            horizontal_flip    = parser.get(VALIDATION, "horizontal_flip", True),
            vertical_flip      = parser.get(VALIDATION, "vertical_flip", True),
            width_shift_range  = parser.get(VALIDATION, "width_shift_range", 0.9), 
            height_shift_range = parser.get(VALIDATION, "height_shift_range", 0.9),
            shear_range        = parser.get(VALIDATION, "shear_range", 0.1), 
            zoom_range         = parser.get(VALIDATION, "zoom_range", 0.1), 
            #brightness_range   = parser.get(VALIDATION, "brightness_range", None)
            )
       else :
          #input("valid_data_augmentation False")
          valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
             rescale            = 1/255,  
             validation_split   = 0.2,
             )
       valid_generator = valid_datagen.flow_from_directory(
             data_dir, 
             target_size   = target_size, 
             batch_size    = 1, #batch_size, # 1 
             interpolation = "bilinear",
             subset        = "validation",
             seed          = seed, 
             shuffle       = False)

    else:
       print("---- No data_augumentation ")
       train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
          rescale            = 1/255,  
          validation_split   = 0.2)
       train_generator = train_datagen.flow_from_directory(
             data_dir, 
             target_size   = target_size, 
             batch_size    = batch_size,
             interpolation = "bilinear",
             subset        = "training",
             seed          = seed, 
             shuffle       = True)

       valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
          rescale            = 1/255,  
          validation_split   = 0.2)
       valid_generator = valid_datagen.flow_from_directory(
             data_dir, 
             target_size   = target_size, 
             batch_size    = 1, #batch_size,  #1  2022/08/12 
             interpolation = "bilinear",
             subset        = "validation", 
             seed          = seed,
             shuffle       = False)

    return (train_generator, valid_generator)


