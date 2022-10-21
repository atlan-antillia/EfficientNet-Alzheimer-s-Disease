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

# 2022/07/20 Copyright (C) antillia.com

# EfficientNetV2Inferencer.py

from operator import ge
import os
import sys
sys.path.append("../../")

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import time
import numpy as np
import glob

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf

import preprocessing

sys.path.append("../../")

from FineTuningModel import FineTuningModel
#from ConfusionMatrix import ConfusionMatrix

import tensorflow as tf

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

FLAGS = flags.FLAGS


def define_flags():
  """Define all flags for binary run."""
  flags.DEFINE_string('mode', 'eval', 'Running mode.')
  flags.DEFINE_string('image_path', None, 'Location of test image.')
  flags.DEFINE_string('label_map',  './label_map.txt', 'Label map txt file')
  flags.DEFINE_integer('image_size', None, 'Image size.')
  flags.DEFINE_string('model_dir', None, 'Location of the checkpoint to run.')
  flags.DEFINE_string('model_name', 'efficientnetv2-s', 'Model name to use.')
  flags.DEFINE_string('dataset_cfg', 'Imagenet', 'dataset config name.')
  flags.DEFINE_string('hparam_str', '', 'k=v,x=y pairs or yaml file.')
  flags.DEFINE_bool('debug', False, 'If true, run in eager for debug.')
  flags.DEFINE_string('export_dir', None, 'Export or saved model directory')
  flags.DEFINE_string('trace_file', '/tmp/a.trace', 'If set, dump trace file.')

  # 2022/07/20
  flags.DEFINE_integer('eval_image_size', None, 'Image size.')
  # 2022/08/14
  flags.DEFINE_float('dropout_rate',  0.3, 'Dropout rate.')

  flags.DEFINE_string('strategy', 'gpu', 'Strategy: tpu, gpus, gpu.')
  flags.DEFINE_integer('num_classes', 10, 'Number of classes.')
  flags.DEFINE_string('best_model_name', 'best_model.5h', 'Best model name.')
  flags.DEFINE_bool('mixed_precision', True, 'If True, use mixed precision.')
  flags.DEFINE_bool('fine_tuning', True,  'Fine tuning flag')
  flags.DEFINE_float('trainable_layers_ratio',  0.3, 'Trainable layers ratio')
  flags.DEFINE_string('infer_dir',  "./inference", 'Directoroy to save inference results.')
  flags.DEFINE_bool('channels_first', False, 'Channel first flag.')
  flags.DEFINE_string('ckpt_dir', "", 'Pretrained checkpoint dir.')


class EfficientNetV2Inferencer:
  # Constructor
  def __init__(self):

    self.classes = []
    #Brain-Tumor-Classification-Dataset
    # https://github.com/sartajbhuvaji/brain-tumor-classification-dataset
    #['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

    with open(FLAGS.label_map, "r") as f:
       lines = f.readlines()
       for line in lines:
        line = line.strip()
        if len(line) >0:
          self.classes.append(line)
    print("--- classes {}".format(self.classes))

    tf.keras.backend.clear_session()

    tf.config.run_functions_eagerly(FLAGS.debug)
  
    model_name  = FLAGS.model_name
    image_size  = FLAGS.image_size
    #num_classes = FLAGS.num_classes
    num_classes = len(self.classes)
    fine_tuning = FLAGS.fine_tuning
    trainable_layers_ratio = FLAGS.trainable_layers_ratio

    if trainable_layers_ratio < 0.1 or trainable_layers_ratio >=0.5:
       print("--- Set default trainable_layers_ratio=0.3")
       trainable_layers_ratio = 0.3
    
    finetuning_model = FineTuningModel(model_name, None, FLAGS.debug)

    self.model = finetuning_model.build(image_size, 
                                        num_classes, 
                                        fine_tuning, 
                                        trainable_layers_ratio = trainable_layers_ratio)
    if FLAGS.debug:
      self.model.summary()
 
    best_model = FLAGS.model_dir + "/best_model.h5"

    if not os.path.exists(best_model):
      raise Exception("Not found best_model " + best_model)
    self.model.load_weights(best_model, by_name=True, )
    print("--- loaded weights {}".format(best_model))
  

  def infer(self ):
    infer_dir   = FLAGS.infer_dir
    if not os.path.exists(infer_dir):
      os.makedirs(infer_dir)
    inference_results_file = os.path.join(infer_dir, "inference.csv")

    NL = "\n"
    SP = ","
    with open(inference_results_file, "w") as f:
      head = "filename, label, prediction1, score1, prediction2, score2" + NL
      f.write(head)

      # image.path = ./somewhere/*.jpg
      image_files = glob.glob(FLAGS.image_path)
      #print(" {}".format(image_files))
      print("--- eval_image_size {}".format(FLAGS.eval_image_size))
      print("\n--- image_path {}".format(FLAGS.image_path))
      image_size  = FLAGS.eval_image_size

      for image_file in image_files:
        image = tf.keras.preprocessing.image.load_img(image_file, target_size=(image_size, image_size),
            color_mode = 'rgb',
            interpolation='nearest')

        image = tf.keras.preprocessing.image.img_to_array(image)
        image = (image)* 1.0/255.0

        # A tensor with a length 1 axis inserted at index axis.
        image  = tf.expand_dims(image, 0)
        logits = self.model(image, training=False)

        pred   = tf.keras.layers.Softmax()(logits)
        idx    = tf.argsort(logits[0])[::-1][:5].numpy()
        basename = os.path.basename(image_file)
        clsname = ""
        try:
          clsname = basename.split("___")[0]
        except:
          pass
        if len(clsname) >0:
          print("\n--- image_file: {}  class: {} ".format(basename, clsname))
        else:
          print("\n--- image_file: {}".format(clsname, basename))

        line = basename + SP + clsname  


        TOP2 = 2
        for i, id in enumerate(idx):
          #print(f'top {i+1} ({pred[0][id]*100:.1f}%):  {self.classes[id]} ')
          score = round(float(pred[0][id]), 4)
          label = self.classes[id] 
          line = line + SP + label + SP + str(score) 

          print(f'prediction {i+1} ({pred[0][id]*100:.1f}%):  {self.classes[id]} ')
          if (i + 1) ==TOP2: 
            break
        #print(line)
        f.write(line + NL)
        

def main(_):
  
  inferncer = EfficientNetV2Inferencer()
  inferncer.infer()


if __name__ == '__main__':
  logging.set_verbosity(logging.ERROR)
  define_flags()
  app.run(main)
