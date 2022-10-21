# Copyright 2022 antillia.com Toshiyuki Arai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ConfigParser.py
#
import os
import sys
import glob
import json
#from collections import OrderedDict
import pprint
import configparser 
import traceback

#Comments were taken from main.py

class ConfigParser:

  # Constructor
  # 
  def __init__(self, config_path):
    print("==== ConfigParser {}".format(config_path))
    if not os.path.exists(config_path):
      raise Exception("Not found config_path {}".format(config_path))

    try:
      self.parse(config_path)
      self.dump_all()
    except Exception as ex:
      print("==== ConfigParser Exception -----------------------{}".format(ex))
      
      traceback.print_exc()


  def parse(self, config_path):
    config = configparser.ConfigParser()
    config.read(config_path)
    self.dict = {s: {i[0]: i[1] for i in config.items(s)}
               for s in config.sections()}

    
  def dump_all(self):

    pprint.pprint(self.dict)


  def get(self, section, name, default_value=None):
    value = None
    try:
      value = self.dict[section][name]
      #print(value)
      value = eval(value)
      
    except:
      traceback.print_exc()
      value = default_value   
    return value


if __name__ == "__main__":
  try:
    file = "trapezoider.ini"
    parser = ConfigParser(file)
    
    n = parser.get("ws_list")
    print(n)
  except:
    traceback.print_exc()
