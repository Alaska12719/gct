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
    data = data[~data['diagnosis_desc'].isnull()]
    data = data[~(data['treat_result'].isnull()) & ~(data['treat_result'] == '其他') & ~(data['treat_result'] == '未治')]
    data['treat_result'].replace('治愈', '0', inplace=True)
    data['treat_result'].replace('好转', '0', inplace=True)
    data['treat_result'].replace('无效', '1', inplace=True)
    data['treat_result'].replace('死亡', '1', inplace=True)
    
    itemcount = {}
    item2id = {}
    id2item = {}
    maxid = 0
    data['item2id'] = '[]'
    rowlen = data.shape[0]
    def retreat(x):
        item = re.sub(r"(（[\u4e00-\u9fa5|\w|\d|\s|-]*)([^\u4e00-\u9fa5]+)([\u4e00-\u9fa5|\w|\d|\s|-]*）)", "\g<1> \g<3>", x)
        ilist = re.findall(r"[\u4e00-\u9fa5|\w|\d|（|）|\s|-]+", item.strip())
        return ilist
    data['item'] = data['diagnosis_desc'].apply(retreat)
    count = 0
    for index, row in data.iterrows():
        matchlist = {}
        ilist = data['item'][index]
        for i in ilist:
            sim_id = sim_in(i, item2id,score)
            if sim_id == -1:
                itemcount[i] = 1
                item2id[i] = maxid
                id2item[maxid] = i
                matchlist[i] = maxid
                maxid += 1
            else:
                sim_str = id2item[sim_id]
                itemcount[sim_str] += 1
                matchlist[sim_str] = sim_id

        data.loc[index, 'item2id'] = repr(matchlist)

        if count % 50 == 0:
            print("\r", end="")
            print("process: %.2f %%" % (count/rowlen * 100), end="")
            sys.stdout.flush()
        count += 1
    print("\r", end="")
    print("process: 100.00 %")

    del data['diagnosis_no']
    del data['diagnosis_type']
    del data['code_version']
    data.to_csv(Path(output_path) / "diagnose.csv", index=False, encoding=model_data_encode)
    with open(Path(output_path) / "diag2id.txt", "w", encoding=model_data_encode) as f:
        f.write(repr(item2id))
    with open(Path(output_path) / "id2diag.txt", "w", encoding=model_data_encode) as f:
        f.write(repr(id2item))

if __name__ == "__main__":
    # 1.创建解释器
    parser = argparse.ArgumentParser(description="Preprocess the DIAGNOSIS dataa, please input the DIAGNOSIS data path.")
    # 2.添加需要的参数
    parser.add_argument('-i', '--input', type=str, required=True, help="the DIAGNOSIS data path")
    parser.add_argument('-o', '--output', type=str, default="./", help="the preprocessed DIAGNOSIS data path")
    parser.add_argument('-e', '--encode', type=str, default="gbk", help="the encode of DIAGNOSIS data")
    parser.add_argument('-s', '--score', type=float, default=0.8, help="the threshold score for diagnosis merge and code filling")
    args = parser.parse_args()
    preprocess(args.input, args.output, args.encode, args.score)