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

# 2022/08/05 Copyright (C) antillia.com

# ConfusionMatrix.py

from fileinput import filename
import os
import sys

from sklearn.metrics import confusion_matrix

import seaborn as sns
#import pandas as pd 
import matplotlib.pyplot as plt

class ConfusionMatrix:
  def __init__(self, 
              fig_size = (8, 6), 
              fmt      = 'd', 
              title_fontsize = 14, 
              label_fontsize = 12, 
              x_ticklabels_rotation = 0, 
              y_ticklabels_rotation = 0, 
              cmap   = 'Blues',
              title  = 'Confution Matrix',
              xlabel = 'Predicted',
              ylabel = 'Actual',
              save_filename= 'confusion_matrix.png'):
    self.fig_size = fig_size
    self.fmt      = fmt
    self.title    = title  #'Confusion Matrix'
    self.xlabel   = xlabel #'Predicted'
    self.ylabel   = ylabel #'Actual'
    self.title_fontsize = title_fontsize
    self.label_fontsize = label_fontsize

    self.x_ticklabels_rotation = x_ticklabels_rotation
    self.y_ticklabels_rotation = y_ticklabels_rotation
    self.cmap     = cmap
    self.save_filename = save_filename


  def create(self, truth, predictions, classes, save_dir):
    labels = sorted(list(set(truth)))
    print("--- truth:\n {}".format(truth))
    cmatrix = confusion_matrix(truth, predictions, labels = labels) 
    print("--- confusion matrix:\n {}".format(cmatrix))
    fig = plt.figure(figsize = self.fig_size) 
    fig.tight_layout()

    ax = sns.heatmap(cmatrix, annot = True, fmt = self.fmt, xticklabels = classes, yticklabels = classes, cmap = self.cmap)

    ax.set_title(self.title, fontsize = self.title_fontsize, weight = 'bold', pad = 20)

    ax.set_xlabel(self.xlabel, fontsize = self.label_fontsize, weight = 'bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation = self.x_ticklabels_rotation)
    ax.set_ylabel(self.ylabel, fontsize = self.label_fontsize, weight = 'bold') 
    ax.set_yticklabels(ax.get_yticklabels(), rotation = self.y_ticklabels_rotation)

    if not os.path.exists(save_dir):
      os.makedirs(save_dir)

    confusion_matrix_file = os.path.join(save_dir, self.save_filename)
    plt.savefig(confusion_matrix_file)    
    #plt.show()
