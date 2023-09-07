from __future__ import division
from __future__ import print_function
from statistics import mode

import graph_convolutional_transformer as gct
import sys
import json
import os
# from Prompt import Prompt

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
      task_type=config["task_type"], #修改这里 prompt，pretrain
      label_key=config["label_key"],
      reg_coef=1.5,
      vocab_sizes={'dx_ints':config['diag_voclen'], 'proc_ints':config['order_voclen']},
      learning_rate=0.00005,
      max_num_codes={'dx':config['diag_max_num'],'px':config['oder_max_num']},
      batch_size=32,
      epoches=config['epoches'],
      id_2_diag_dir=id_2_diag_dir,
      id_2_order_dir=id_2_order_dir,
      use_bert=config["use_bert"],
      use_position=config["use_position"],
      fine_tune_bert=config["fine_tune_bert"],
      bert_epoches=config["bert_epoches"],
      use_attention=False,#default not modify
      encode=config['model_data_encode'],
      num_classes=config["num_class"],#four classifier
      input_path=input_path,)
  #pretrain + prompt
  model.run_gct_pretrain(model_dir, config["task_type"])
  # model.run_gct_pretrain(model_dir, 'prompt') #如果使用fine_tune需要将参数改为fine_tune
  # model.run_gct_pretrain(model_dir, 'fine_tune')
if __name__ == '__main__':
  main(sys.argv)
