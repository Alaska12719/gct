import pandas as pd
import numpy as np
import os
import re
import sys
from pathlib import Path
import difflib
import argparse


def string_similar(s1, s2):
    return difflib.SequenceMatcher(None, s1, s2).quick_ratio()

def sim_in(str, iterlist, p):
    for i in iterlist:
        v = string_similar(str,i)
        if v > p:
            return iterlist[i]
    return -1

def preprocess(input_path, output_path, encode, model_data_encode, score):
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    data = pd.read_csv(input_path, encoding=encode, low_memory=False)
    data = data[~data['order_text'].isnull()]
    rowlen = data.shape[0]
    item2id = {}
    id2item = {}
    count = 0
    notnulldata = data[~data['order_code'].isnull()]
    for i,row in notnulldata.iterrows():
        item2id[data['order_text'][i]] = data['order_code'][i]
        id2item[data['order_code'][i]] = data['order_text'][i]
        if count % 50 == 0:
            print("\r", end="")
            print("process: %.2f %%" % (count/rowlen * 100), end="")
            sys.stdout.flush()
        count += 1
    
    nullcodedata = data[data['order_code'].isnull()]
    # def treat(datarow):
    #     if datarow['order_text'] in item2id:
    #         datarow['order_code'] = item2id[datarow['order_text']]
    #         if count % 50 == 0:
    #             print("\r", end="")
    #             print("process: %.2f %%" % (count/rowlen * 100), end="")
    #             sys.stdout.flush()
    #         count += 1
    #     return datarow
    # nullcodedata.apply(treat, axis=1)

    for i,row in nullcodedata.iterrows():
        if data['order_text'][i] in item2id:
            data.loc[i, 'order_code'] = item2id[data['order_text'][i]]
            if count % 50 == 0:
                print("\r", end="")
                print("process: %.2f %%" % (count/rowlen * 100), end="")
                sys.stdout.flush()
            count += 1
    
    nulldata = data[data['order_code'].isnull()]
    maxid = 0
    for i,row in nulldata.iterrows():
        desc = nulldata['order_text'][i]
        sim_id = sim_in(desc, item2id, score)
        if sim_id == -1:
            item2id[desc] = "AI{}".format(maxid)
            id2item["AI{}".format(maxid)] = desc
            data.loc[i, 'order_code'] = "AI{}".format(maxid)
            maxid += 1
        else:
            data.loc[i, 'order_text'] = id2item[sim_id]
            data.loc[i, 'order_code'] = sim_id
        
        if count % 50 == 0:
            print("\r", end="")
            print("process: %.2f %%" % (count/rowlen * 100), end="")
            sys.stdout.flush()
        count += 1
    print("\r", end="")
    print("process: 100.00 %")

    datacount = data.count()
    rowcount = data.shape[0]
    for x in range(datacount.shape[0]):
        i = datacount.index[x]
        if datacount[i] < rowcount/2:
            del data[i]
    
    data.to_csv(Path(output_path) / "order.csv", index=False, encoding=model_data_encode)
    with open(Path(output_path) / "order2id.txt", "w", encoding=model_data_encode) as f:
        f.write(repr(item2id))
    with open(Path(output_path) / "id2order.txt", "w", encoding=model_data_encode) as f:
        f.write(repr(id2item))

if __name__ == "__main__":
    # 1.创建解释器
    parser = argparse.ArgumentParser(description="Preprocess the ORDER dataa, please input the ORDER data path.")
    # 2.添加需要的参数
    parser.add_argument('-i', '--input', type=str, required=True, help="the ORDER data path")
    parser.add_argument('-o', '--output', type=str, default="./", help="the preprocessed ORDER data path")
    parser.add_argument('-e', '--encode', type=str, default="gbk", help="the encode of ORDER data")
    parser.add_argument('-s', '--score', type=float, default=0.8, help="the threshold score for ORDER merge and code filling")
    args = parser.parse_args()
    preprocess(args.input, args.output, args.encode, args.score)