# layer normalization
import torch
from torch import nn
import re
import torch
import utils
import sys
import copy
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
try:
    path = sys.argv[1]
except IndexError as e:
    print(repr(e), "\n\nplease run the code with a config file, for example:\n     python preprocess.py config.json")
    exit(0)
def search(file):
    count = 0
    result = {}
    ds = {}
    with open(file) as f:
        for line in f:
            count = (count + 1) % 13
            if count == 4 :
                ds["valid_col0"] = float(line)
            if count == 6 :
                ds["valid_col1"] = float(line)
            if count == 7 :
                ds["valid_loss"] = float(line.split("loss ")[1].split(",")[0])
                ds["valid_flscore_micro"] = float(line.split("flscore_micro: ")[1].split(",")[0])
                ds["valid_flscore_macro"] = float(line.split("flscore_macro: ")[1].split(",")[0])
                ds["valid_flscore_weighted"] = float(line.split("flscore_weighted: ")[1].split(",")[0])
                ds["valid_flscore_samples"] = float(line.split("flscore_samples: ")[1].split(",")[0].split("\n")[0])
            if count == 10 :
                ds["test_col0"] = float(line)
            if count == 12 :
                ds["test_col1"] = float(line)
            if count == 0 :
                ds["test_loss"] = float(line.split("loss ")[1].split(",")[0])
                ds["test_flscore_micro"] = float(line.split("flscore_micro: ")[1].split(",")[0])
                ds["test_flscore_macro"] = float(line.split("flscore_macro: ")[1].split(",")[0])
                ds["test_flscore_weighted"] = float(line.split("flscore_weighted: ")[1].split(",")[0])
                ds["test_flscore_samples"] = float(line.split("flscore_samples: ")[1].split(",")[0].split("\n")[0])
                ds["epoch"] = int(line.split("epoch ")[1].split(" loss")[0])
                # print(int(line.split("epoch ")[1].split(" loss")[0]))
                result[int(line.split("epoch ")[1].split(" loss")[0])] = copy.deepcopy(ds)
                ds = {}
    max = 1
    for i in range(1,len(result)):
        if result[i]['valid_flscore_macro'] > result[max]['valid_flscore_macro']:
            max = i
    return result[max]
    
print(search(path))
