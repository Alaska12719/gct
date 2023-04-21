"""Copyright 2019 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import division
from __future__ import print_function
from statistics import mode

import graph_convolutional_transformer as gct
import sys
import json
import os

try:
    path = sys.argv[1]
except IndexError as e:
    print(repr(e), "\n\nplease run the code with a config file, for example:\n     python preprocess.py config.json")
    exit(0)

with open(path, "r") as f:
    config = json.load(f)
input_path = config["output_path"]
model_dir = config["model_dir"]
if not os.path.exists(model_dir):
  os.mkdir(model_dir)
id_2_diag_dir = input_path +"/id2diag_new.txt"
id_2_order_dir = input_path +"/id2oder_new.txt"

def main(argv):
  
  gct_params = {
      "embedding_size": 768,
      "num_transformer_stack": 3,
      "num_feedforward": 2,
      "num_attention_heads": 1,
      "ffn_dropout": 0.72,
      "attention_normalizer": "softmax",
      "multihead_attention_aggregation": "concat",
      "directed_attention": False,
      "use_inf_mask": config["use_inf_mask"],
      "use_prior": config["use_prior"],
      "training": config["training"],
  }


  model = gct.EHRTransformer(
      gct_params=gct_params,
      label_key=config["label_key"],
      reg_coef=1.5,
      vocab_sizes={'dx_ints':config['diag_voclen'] + 1, 'proc_ints':config['order_voclen'] + 1},
      learning_rate=0.00005,
      max_num_codes={'dx':config['diag_max_num'] + 1,'px':config['oder_max_num'] + 1},
      batch_size=32,
      epoches=config['epoches'],
      id_2_diag_dir=id_2_diag_dir,
      id_2_order_dir=id_2_order_dir,
      use_bert=config["use_bert"],
      use_attention=False,#default not modify
      encode=config['model_data_encode'],
      num_classes=config["num_class"],#four classifier
      input_path=input_path,)
  model.run_gct_model(model_dir)
if __name__ == '__main__':
  main(sys.argv)
