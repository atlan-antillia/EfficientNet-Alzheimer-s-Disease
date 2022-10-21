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

# ClassificationWriter.py

import os
import sys
import json
import pprint
from sklearn.metrics import classification_report

import pandas as pd 

class ClassificationReportWriter:

  def __init__(self, cls_report_filename= "classification_report.csv"):
    self.CSV    = ".csv"
    self.JSON   = ".json"
    self.cls_report_filename = cls_report_filename


  def write(self, truth, predictions, classes, save_dir):
    print("--- ClassificationReportWriter.write: classes{}".format(classes))
    cls_report = classification_report( 
                               y_true       = truth,
                               y_pred       = predictions,
                               target_names = classes, 
                               output_dict  = True)
    print("--- classification report:\n")
    pprint.pprint(cls_report)

    if self.cls_report_filename.endswith(self.CSV):
      report_df = pd.DataFrame(cls_report).T
      cls_report_csv_file = os.path.join(save_dir, self.cls_report_filename)

      report_df.to_csv(cls_report_csv_file)
      print("--- Saved classification_report file {}".format(cls_report_csv_file))
    elif self.cls_report_filename.endswith(self.JSON):
      cls_report_json_file = os.path.join(save_dir, self.cls_report_filename)

      with open(cls_report_json_file, 'w', encoding='utf-8') as f:
        json.dump(cls_report, f, ensure_ascii=False, indent=4)
    else:
      print("Unsuppored cls_report_filename type {}".format(self.cls_report_filename))
