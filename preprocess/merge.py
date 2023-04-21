import pandas as pd
import numpy as np
from random import randint, sample
from sklearn.model_selection import train_test_split 
import sys
import os
from pathlib import Path
import argparse


def preprocess(input_path, output_path, model_data_encode, conditions, sample_num):         
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    input_path = Path(input_path)
    output_path = Path(output_path)
    print("[0/8]: load the processed diagnosis and order data")
    sys.stdout.flush()
    diag = pd.read_csv(input_path / "diagnose.csv", low_memory=False, encoding=model_data_encode)
    oper = pd.read_csv(input_path / "order.csv", low_memory=False, encoding=model_data_encode)

    # filter the orders
    order_class = oper['order_class']
    print("[1/8]: filter the orders according to order class")
    sys.stdout.flush()
    if conditions: 
        generate_condition = False
        for i in conditions:
            generate_condition = generate_condition | (order_class == i)
        oper = oper[generate_condition]
    # oper = oper[order_class >= 'A' & order_class <= 'F' & order_class != 'E']

    # merge diagnosis
    print("[2/8]: merge diagnosis from one visit")
    sys.stdout.flush()
    diagdict = {}
    for i, row in diag.iterrows():
        id = row['patient_id'] + '_' + str(row['visit_id'])
        if id in diagdict:
            value = diagdict[id]
            diagdict[id] = dict(value, **eval(row['item2id']))
        else:
            diagdict[id] = eval(row['item2id'])

    # merge orders
    print("[3/8]: merge orders from one visit")
    sys.stdout.flush()
    oderdict = {}
    for i, row in oper.iterrows():
        id = row['patient_id'] + '_' + str(row['visit_id'])
        if id not in oderdict:
            oderdict[id] = {}
        oderdict[id][row['order_text']] = row['order_code']

    print("[4/8]: choose the visits that have same label for each diagnosis")
    sys.stdout.flush()
    # choose the single label
    labeldict = {}
    for i, row in diag.iterrows():
        label = row['treat_result']
        id = row['patient_id'] + '_' + str(row['visit_id'])
        if id in labeldict:
            value = labeldict[id]
            labeldict[id].add(label)
        else:
            labeldict[id] = {label}

    # delete multilabel rows
    multilabel = {}
    sublabel = set()
    for i in labeldict:
        if len(labeldict[i]) > 1:
            multilabel[i] = labeldict[i]
        if i not in oderdict:
            sublabel.add(i)

    singlelabel = {key:labeldict[key] for key in (labeldict.keys() - multilabel.keys() - sublabel)}
    print("[5/8]: merge the data of diagnosis and orders")
    sys.stdout.flush()
    all_data = {}
    for id in singlelabel:
        newoderdict = {}
        if sample_num is not None and len(oderdict[id]) > sample_num:
            samplekey = sample(list(oderdict[id]), sample_num)
            for i in samplekey:
                newoderdict[i] = oderdict[id][i]
        else:
            newoderdict = oderdict[id]
        all_data[id] = {
            'diagnose': diagdict[id],
            'order': newoderdict,
            # 'order': oderdict[id],
            'label': list(singlelabel[id])[0]
        }

    with open( output_path / "vid2diag.txt", "w", encoding=model_data_encode) as f:
        f.write(repr(diagdict))
    with open( output_path / "vid2oder.txt", "w", encoding=model_data_encode) as f:
        f.write(repr(oderdict))
    with open( output_path / "all_data.txt", "w", encoding=model_data_encode) as f:
        f.write(repr(all_data))

    print("[6/8]: construct conditional matrix P")
    sys.stdout.flush()
    #p(d)
    diagp = {}
    total = 0
    for i in all_data:
        diag = all_data[i]['diagnose']
        for d in diag:
            if d not in diagp:
                diagp[d] = 1
            else:
                diagp[d] += 1
        total += 1
    for i in diagp:
        diagp[i] = diagp[i]/total

    #p(o)
    operp = {}
    for i in all_data:
        oper = all_data[i]['order']
        for o in oper:
            if o not in operp:
                operp[o] = 1
            else:
                operp[o] += 1
    for i in operp:
        operp[i] = operp[i]/total


    # re-encoding
    print("[7/8]: re-encoding the diagnosis and order node jointly")
    sys.stdout.flush()
    cid2diag = {}
    cdiag2id = {}
    cdiagp = {}
    id = 0
    for i in diagp:
        cid2diag[id] = i
        cdiag2id[i] = id
        cdiagp[id] = diagp[i]
        id += 1

    cid2oper = {}
    coper2id = {}
    coperp = {}
    id = 0
    for i in operp:
        cid2oper[id] = i
        coper2id[i] = id
        coperp[id] = operp[i]
        id += 1

    cdop = {}

    for vid in all_data:
        diag = all_data[vid]['diagnose']
        oper = all_data[vid]['order']
        for d in diag:
            for o in oper:
                id = "{}_{}".format(cdiag2id[d], coper2id[o])
                if id in cdop:
                    cdop[id] += 1
                else:
                    cdop[id] = 1

    for id in cdop:
        cdop[id] /= total

    # cal
    conp_do = {}
    conp_od = {}
    for d in cdiagp:
        for o in coperp:
            id = "{}_{}".format(d,o)
            if id in cdop:
                conp_do[id] = cdop[id] / coperp[o]
                rid = "{}_{}".format(o,d)
                conp_od[rid] = cdop[id] / cdiagp[d] if id in cdop else 0
        
    print("[8/8]: write data to the disk")
    sys.stdout.flush()
    with open(output_path / "diag2id_new.txt", "w", encoding=model_data_encode) as f:
        f.write(repr(cdiag2id))
    with open(output_path / "id2diag_new.txt", "w", encoding=model_data_encode) as f:
        f.write(repr(cid2diag))
    with open(output_path / "oder2id_new.txt", "w", encoding=model_data_encode) as f:
        f.write(repr(coper2id))
    with open(output_path / "id2oder_new.txt", "w", encoding=model_data_encode) as f:
        f.write(repr(cid2oper))
    with open(output_path / "diagp_new.txt", "w", encoding=model_data_encode) as f:
        f.write(repr(cdiagp))
    with open(output_path / "oderp_new.txt", "w", encoding=model_data_encode) as f:
        f.write(repr(coperp))
    with open(output_path / "conp_do_new.txt", "w", encoding=model_data_encode) as f:
        f.write(repr(conp_do))
    with open(output_path / "conp_od_new.txt", "w", encoding=model_data_encode) as f:
        f.write(repr(conp_od))
    with open(output_path / "cdop_new.txt", "w", encoding=model_data_encode) as f:
        f.write(repr(cdop))

    for vid in all_data:
        diag = all_data[vid]['diagnose']
        oper = all_data[vid]['order']
        for i in diag:
            all_data[vid]['diagnose'][i] = cdiag2id[i]
        for i in oper:
            all_data[vid]['order'][i] = coper2id[i]
    with open(output_path / "all_data_new.txt", "w", encoding=model_data_encode) as f:
        f.write(repr(all_data))

    def getmaxlen(attr):
        maxlen = 0
        for i in all_data:
            if maxlen < len(all_data[i][attr]):
                maxlen = len(all_data[i][attr])
        return maxlen

    order_maxlen = getmaxlen('order')
    diag_maxlen = getmaxlen('diagnose')
    diag_voclen = len(cdiag2id)
    order_voclen = len(coper2id)
    
    dc = all_data
    d2o = conp_do
    o2d = conp_od
    df = pd.DataFrame(columns=["patientId", 'label.expired', 'label.readmission','dx_ids','dx_ints','proc_ids','proc_ints','prior_indices','prior_values'])
    for patientId in dc:
        l=[]
        l.append(patientId) #patientId
        # readmission and expired label
        l.append(dc[patientId]['label'])     
        l.append(dc[patientId]['label'])
        # dx_ids
        l.append(list(dc[patientId]['diagnose'].keys()))
        # dx_ints
        l.append(list(dc[patientId]['diagnose'].values()))
        # proc_ids
        l.append(list(dc[patientId]['order'].keys()))
        # proc_ints
        l.append(list(dc[patientId]['order'].values()))
        # proc metrix
        dx_ints = list(dc[patientId]['diagnose'].values())
        proc_ints = list(dc[patientId]['order'].values())
        prior_indices = []
        prior_values = []
        for i in range(len(dx_ints)):
            for j in range(len(proc_ints)):
                key = str(dx_ints[i]) + '_' + str(proc_ints[j])
                prior_indices.append(i)
                prior_indices.append(diag_maxlen + j)
                prior_values.append(d2o[key])

        for i in range(len(proc_ints)):
            for j in range(len(dx_ints)):
                key = str(proc_ints[i]) + '_' + str(dx_ints[j])
                prior_indices.append(diag_maxlen + i)
                prior_indices.append(j)
                prior_values.append(o2d[key])
        l.append(prior_indices)
        l.append(prior_values)
        df.loc[len(df)] = l
    data_train, data_test = train_test_split(df, test_size=0.2, random_state=1234)
    # split validate set and test set
    data_test, data_val = train_test_split(data_test, test_size=0.5, random_state=1234)

    data_train.to_csv(output_path / "train.csv", index=False, encoding=model_data_encode)
    data_val.to_csv(output_path / "validation.csv", index=False, encoding=model_data_encode)
    data_test.to_csv(output_path / "test.csv", index=False, encoding=model_data_encode)

    return diag_maxlen, order_maxlen, diag_voclen, order_voclen

if __name__ == "__main__":
    # 1.创建解释器
    parser = argparse.ArgumentParser(description="Preprocess the ORDER dataa, please input the ORDER data path.")
    # 2.添加需要的参数
    parser.add_argument('-i', '--input', type=str, required=True, help="the input data dir path")
    parser.add_argument('-o', '--output', type=str, default="./", help="the preprocessed merge data path")
    parser.add_argument('-c', '--conditions', type=str, default=None, help="the condition of order class")
    parser.add_argument('-n', '--sample_num', type=int, default=None, help="the sample number of order")
    args = parser.parse_args()
    preprocess(args.input, args.output, args.conditions, args.sample_num)