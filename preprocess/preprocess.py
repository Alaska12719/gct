import pre_diagnose
import pre_order
import merge
import json
import sys
from pathlib import Path

try:
    path = sys.argv[1]
except IndexError as e:
    print(repr(e), "\n\nplease run the code with a config file, for example:\n     python preprocess.py config.json")
    exit(0)

with open(path, "r") as f:
    config = json.load(f)

# preprocess diagnose
print("=====preprocess diagnose=====")
sys.stdout.flush()
diag_path = Path(config['input_path']) / config['diagnosis_name']
pre_diagnose.preprocess(diag_path, config['output_path'], config['encode'], config['model_data_encode'], config['score'])

# preprocess orders
print("=====preprocess orders=====")
sys.stdout.flush()
order_path = Path(config['input_path']) / config['orders_name']
pre_order.preprocess(order_path, config['output_path'], config['encode'], config['model_data_encode'], config['score'])

# build train, test, valid dataset
print("=====build train, test, valid dataset=====")
sys.stdout.flush()
conditions = None if config['conditions'] == "None" else eval(config['conditions'])
sample_num = None if config['sample_num'] == -1 else config['sample_num']
diag_maxlen, order_maxlen, diag_voclen, order_voclen = merge.preprocess(config['output_path'], config['output_path'], config['model_data_encode'], conditions, sample_num)
config['diag_max_num'] = diag_maxlen
config['oder_max_num'] = order_maxlen
config['diag_voclen'] = diag_voclen
config['order_voclen'] = order_voclen

print("=====Well done! The processed data all in {}=====".format(config['output_path']))
sys.stdout.flush()

with open(path, 'w') as f:
    json.dump(config, f)