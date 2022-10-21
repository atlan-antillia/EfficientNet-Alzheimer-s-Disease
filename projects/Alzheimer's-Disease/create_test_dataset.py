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

# 2022/10/22 Copyright (C) antillia.com

import os
import sys
import glob
import numpy as np
import random
import shutil
import traceback

def create_test_dataset(test_dir, target_dir, num=20):
  classes = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]

  for cls in classes:
   
    dir = test_dir + "/" + cls
    pattern = dir + "/*.jpg"
    files = glob.glob(pattern)
    print("-- files {}".format(files))
    samples = random.sample(files, num)
    for i, sample in enumerate(samples):
       basename = os.path.basename(sample)
       nameonly = basename.split(".")[0]
       target_file = target_dir + "/" + cls + "___" + nameonly + "_" + str(i+101) + ".jpg"
       print("--- copied {} to {}".format(sample, target_file))
       shutil.copy2(sample, target_file)

    
if __name__ == "__main__":
  test_dir = "./Resampled_Alzheimer's_Images/test"
  dest_dir = "./test"
  try:
    if not os.path.exists(test_dir):
      raise Exception("Not found {}".format(test_dir))


    if os.path.exists(dest_dir):
      shutil.rmtree(dest_dir)

    if not os.path.exists(dest_dir):
      os.makedirs(dest_dir)

    create_test_dataset(test_dir, dest_dir, num=10)

  except:
    traceback.print_exc()
