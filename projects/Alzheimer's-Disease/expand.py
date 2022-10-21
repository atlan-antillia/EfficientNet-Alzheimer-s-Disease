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

# 2022/10/21 Copyright (C) antillia.com

# expand.py

import os
import sys
import shutil
import glob
from PIL import Image, ImageDraw
import traceback

class ImageExpander:

  def __init__(self):
    pass

  def expand(self, input_dir, output_dir, target_size=512):
    subdirs = os.listdir(input_dir)
    print(subdirs)
    for subdir in subdirs:
      output_subdir = os.path.join(output_dir, subdir)
      if not os.path.exists(output_subdir):
        os.makedirs(output_subdir) 

      subdir_path = os.path.join(input_dir, subdir) 
      files = glob.glob(subdir_path + "/*.jpg")
      print("--- dir {}  num files {}".format(subdir_path, len(files)))
      for file in files:  
        image = Image.open(file)
        image = image.convert('RGB')

        GRAY = (128, 128, 128)
        background = Image.new(mode = "RGB", 
                           size = (target_size, target_size),
                           color = GRAY )

        w, h = image.size
        #print("filename {} original size w {} h {}".format(file, w, h))
        base_size = w
        if h>w:
          base_size = h
        expanding_ratio = target_size / base_size
        #print("--- expanding ratio {}".format(expanding_ratio))
        rw = float(w) * expanding_ratio
        rh = float(h) * expanding_ratio
        rw = int(rw)
        rh = int(rh)
        (r, g, b)   = image.getpixel((2, 2))
        print("--- original size {} {} to {} {}".format(w, h, rw, rh))
        expanded_image = image.resize((rw, rh))
        draw = ImageDraw.Draw(background)
        draw.rectangle((0, 0, target_size, target_size),
                                         fill=(r, g, b))
        x = (target_size - rw)/2.0
        y = (target_size - rh)/2.0
        x = int(x)
        y = int(y)
        background.paste(expanded_image, (x, y))
        basename = os.path.basename(file)
        filename = basename.split(".")[0]
        output_filepath = os.path.join(output_subdir, filename + ".jpg")
        #background.show()
        #input("hit any key")
        background.save(output_filepath, quality=95)
        
  
if __name__ == "__main__":
  input_dir  = "./Alzheimer's_Images/train"
  output_dir = "./Resized_Alzheimer's_512x512_master/train"
  
  try:
    if not os.path.exists(input_dir):
      raise Excpetion("Not found " + input_dir)
    
    if os.path.exists(output_dir):
      shutil.rmtree(output_dir)
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
    expander = ImageExpander()
    expander.expand(input_dir, output_dir)
  except:
    traceback.print_exc()

  input_dir  = "./Alzheimer's_Images/test"
  output_dir = "./Resized_Alzheimer's_512x512_master/test"
  
  try:
    if not os.path.exists(input_dir):
      raise Excpetion("Not found " + input_dir)
    
    if os.path.exists(output_dir):
      shutil.rmtree(output_dir)
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
    expander = ImageExpander()
    expander.expand(input_dir, output_dir)

  except:
    traceback.print_exc()
