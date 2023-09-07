import argparse
import math

import torch

import sys
from tqdm import tqdm
import pandas as pd
from torch import nn, softmax
from torch.utils.data import Dataset,DataLoader
import numpy as np
import time
from torch import optim
from sklearn.metrics import precision_recall_curve,auc,roc_auc_score,f1_score, roc_curve,accuracy_score, precision_score, recall_score, confusion_matrix
import os
import random
from transformers import AutoTokenizer, AutoModel
from utils import one_hot,adjust_learning_rate
import warnings
from model import SimcseModel,simcse_unsup_loss
from dataset import TrainDataset
warnings.filterwarnings("ignore")
glob_features = []
# torch.cuda.set_device(int(os.environ['CUDA_VISIBLE_DEVICES']))
# torch.cuda.set_device(0)
def seed_it(seed):
    random.seed(seed) #可以注释掉
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) #这个懂吧
    torch.backends.cudnn.deterministic = True #确定性固定
    torch.backends.cudnn.benchmark = True #False会确定性地选择算法，会降低性能
    torch.backends.cudnn.enabled = True  #增加运行效率，默认就是True
    torch.manual_seed(seed)
seed_it(1314)
#注意max_num_codes出现的地方
#_multihead_aggregation没考虑
#may error norm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class PositionalEncoding(nn.Module):
    def __init__(self, dim, dropout,enable, max_len=5000):
        super(PositionalEncoding, self).__init__()

        if dim % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(dim))

        """
        构建位置编码pe
        pe公式为：
        PE(pos,2i/2i+1) = sin/cos(pos/10000^{2i/d_{model}})
        """
        pe = torch.zeros(max_len, dim)  # max_len 是解码器生成句子的最长的长度，假设是 10
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))


        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)
        self.drop_out = nn.Dropout(p=dropout)
        self.dim = dim
        self.position_enable_ = enable

    def forward(self, emb, step=None):
        if(self.position_enable_):
            emb = emb * math.sqrt(self.dim)

            if step is None:
                emb = emb + self.pe[:emb.size(0)]
            else:
                emb = emb + self.pe[step]
            emb = self.drop_out(emb)
            return emb
        return emb

def run_print_result(task_type, epoch,num_class, get_prediction, get_loss, model, feature_embedder, features, linear_net, show_attention = False, save_dir = ".", decoder=None):
    logits, attentions, hidden = get_prediction(
                model, feature_embedder, features, linear_net, training=False)
    tmp = []
    tmp_count = 0
    valid_idx_len = 0
    decode_loss = 0
    predict_decoder = []
    true_decoder = []
    vocab_key = feature_embedder.get_vocab_and_key()
    for key in vocab_key['feature_key']:
        tmp_count = vocab_key['vocab'][key] + tmp_count
    

    if decoder != None:
        for i in range(hidden.shape[0]):
            start_idx = 1
            input_tensor = []
            true_tensor = []
            for key in vocab_key['feature_key']:
                tmp = hidden[i][start_idx:features[key].shape[-1] + start_idx]
                start_idx += features[key].shape[-1]
                valid_idx = torch.nonzero(features[key][i] != vocab_key['vocab'][key]).squeeze()
                if len(valid_idx.shape) == 0:
                    valid_idx_len = 1
                    valid_idx = valid_idx.unsqueeze(0)
                else:
                    valid_idx_len = valid_idx.shape[0]
                input_tensor.append(tmp[valid_idx])
                one_hot_tensor = torch.zeros(valid_idx_len, tmp_count).to(device)
                one_hot_tensor.scatter_(1, (features[key][i][valid_idx] + start_idx -  features[key].shape[-1]).unsqueeze(1), 1)
                true_tensor.append(one_hot_tensor)
            cur_loss, predict_sum, true_sum = decoder(torch.cat(input_tensor).detach(), torch.cat(true_tensor))
            predict_decoder += predict_sum
            true_decoder += true_sum
            decode_loss += cur_loss
            
    if logits.shape[1] > 1:
        nn_softmax = nn.Softmax()
        probs = nn_softmax(logits)
    else:
        nn_sigmoid = nn.Sigmoid()
        probs = nn_sigmoid(logits)

    predictions = {
        'probabilities': probs,
        'logits': logits,
    }
    labels = features['label'].to(device).float()
    if logits.shape[1] > 1:
        labels = one_hot(labels, num_class)
    loss = get_loss(logits, labels, attentions)
    
    out_probs = probs.cpu().detach().numpy()
    out_labels = np.array(labels.cpu())

    # produce attention and prior and id file
    if show_attention == True:
        attentions = attentions[2].cpu().detach().numpy()
        for i in range(len(features['patientId'])):
            dict = {}
            dict['attention'] = attentions[i][0]
            dict['prior'] = features['prior_guide'][i].cpu().detach().numpy()
            dict['proc'] = features['proc_ints'][i].cpu().detach().numpy()
            dict['dx'] = features['dx_ints'][i].cpu().detach().numpy()
            dict['proc_num'] = features['proc_num'][i].cpu().detach().numpy()
            dict['dx_num'] = features['dx_num'][i].cpu().detach().numpy()
            dict['patientId'] = features['patientId'][i]
            np.save(save_dir + '/' + task_type + '/' + features['patientId'][i] + '.npy', dict)

    if task_type != 'train':
        if logits.shape[1] > 1:
            out_probs = np.argmax(out_probs,axis=1)
            out_probs = np.eye(num_class)[out_probs].astype(int)
            return out_probs, out_labels, loss, predict_decoder, true_decoder, decode_loss
        else:
            precision, recall, thresholds = precision_recall_curve(out_labels, out_probs)
            auc_precision_recall = auc(recall, precision)
            roc_auc = roc_auc_score(out_labels, out_probs)
            
            print('epoch %d loss %.5f, epoch AUC: %.5f ,AUCPR: %.5f' % (
                epoch + 1, loss, roc_auc, auc_precision_recall), flush=True)
    if task_type=='train':
        return loss, predict_decoder, true_decoder, decode_loss

def run_model(task_type, epoch,num_class, dataloader, get_prediction, get_loss, optimizer, model, feature_embedder, linear_net, max_epoch = 200, show_attention = False, save_dir='.', decoder=None, decoder_optimizer=None):
    out_labels = []
    out_probs = []
    loss = 0
    for (idx, features) in enumerate(dataloader):
        if task_type == "train":
            loss, predicted_labels, true_labels,decoded_loss  = run_print_result(task_type, epoch,num_class, get_prediction, get_loss, model, feature_embedder, features, linear_net, decoder=decoder)
            if decoder != None:
                calculateDecode(predicted_labels, true_labels)
                decoder_optimizer.zero_grad()
                decoded_loss.backward()
                decoder_optimizer.step()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                out_probs_epoch,out_labels_epoch,loss_epoch, predicted_labels, true_labels, decoded_loss = run_print_result(task_type, epoch,num_class, get_prediction, get_loss, model, feature_embedder, features, linear_net, show_attention, save_dir=save_dir, decoder=decoder)
                out_labels.append(out_labels_epoch)
                if decoder != None:
                    calculateDecode(predicted_labels, true_labels)
                out_probs.append(out_probs_epoch)
                loss += loss_epoch * out_probs_epoch.shape[0]
        
    if task_type != "train":
        print('Task ======== ' + task_type)
        out_labels = np.vstack(out_labels)
        out_probs = np.vstack(out_probs)
        loss /= out_labels.shape[0]
        flscore_micro = f1_score(out_labels, out_probs, average='micro')
        flscore_macro = f1_score(out_labels, out_probs, average='macro')
        flscore_weighted = f1_score(out_labels, out_probs, average='weighted')
        flscore_samples = f1_score(out_labels, out_probs, average='samples')
        for col in range(num_class):
            prob = out_probs[:,col]
            true_label = out_labels[:, col]
            f1 = f1_score(true_label, prob)
            print('==================' + str(col) + ' column ==============')
            print(f1)
        print('epoch %d loss %.5f, flscore_micro: %.5f, flscore_macro: %.5f, flscore_weighted: %.5f, flscore_samples: %.5f' % (
            epoch + 1, loss, flscore_micro, flscore_macro, flscore_weighted, flscore_samples), flush=True)


def run_pretrain_result(task_type, epoch,num_class, get_prediction, get_loss, model, feature_embedder, features, linear_net, show_attention = False, save_dir = ".", decoder=None, decoder_optimizer=None):
    logits,hidden = get_prediction(
                model, feature_embedder, features, linear_net, training=False)
    logits = logits[:,2:]
    vocab_key = feature_embedder.get_vocab_and_key()
    tmp = []
    tmp_count = 0
    valid_idx_len = 0
    decode_loss = 0
    predict_decoder = []
    true_decoder = []
    for key in vocab_key['feature_key']:
        tmp.append(logits[:,tmp_count:vocab_key['vocab'][key] + tmp_count])
        tmp_count = vocab_key['vocab'][key] + tmp_count + 1
    logits = torch.cat(tmp,dim=1).to(device)
    if decoder != None:
        for i in range(hidden.shape[0]):
            start_idx = 2
            input_tensor = []
            true_tensor = []
            for key in vocab_key['feature_key']:
                tmp = hidden[i][start_idx:features[key].shape[-1] + start_idx]
                start_idx += features[key].shape[-1]
                valid_idx = torch.nonzero(features[key][i] != vocab_key['vocab'][key]).squeeze()
                if len(valid_idx.shape) == 0:
                    valid_idx_len = 1
                    valid_idx = valid_idx.unsqueeze(0)
                else:
                    valid_idx_len = valid_idx.shape[0]
                input_tensor.append(tmp[valid_idx])
                one_hot_tensor = torch.zeros(valid_idx_len, tmp_count - len(vocab_key['feature_key'])).to(device)
                one_hot_tensor.scatter_(1, (features[key][i][valid_idx] + start_idx -  features[key].shape[-1] - 2).unsqueeze(1), 1)
                true_tensor.append(one_hot_tensor)
            cur_loss, predict_sum, true_sum = decoder(torch.cat(input_tensor).detach(), torch.cat(true_tensor))
            predict_decoder += predict_sum
            true_decoder += true_sum
            decode_loss += cur_loss
            
    
    
    labels = features['label']
    true_labels = feature_embedder.lookupLabels(labels)
    # num_class = linear_net.get_vocab_size()
    # true_labels = torch.nn.functional.one_hot(true_labels, num_classes=num_class)
    crossLoss = nn.CrossEntropyLoss().to(device)
    softmax = nn.Softmax(dim=1).to(device)
    loss = crossLoss(logits.float().to(device), true_labels.to(device))
    true_labels = true_labels.to(device)
    origin_feature = []
    if task_type != "train":
        for i in range(true_labels.shape[0]):
            origin_feature.append("patientId = " + features['patientId'][i] + ", diagnose = " + features['dx_ids'][i] + ", proc = " +  features['proc_ids'][i])
            origin_feature.append("true label is " + glob_features[true_labels[i]] + ", predict label is " + glob_features[softmax(logits).argmax(dim=1)[i]])
        return origin_feature, (true_labels == softmax(logits).argmax(dim=1)).sum(),true_labels.shape[0], loss
    # if task_type=='train':
    print(task_type + 'Task ======== Loss:' + str(loss.item()))
    return loss, decode_loss, predict_decoder, true_decoder
        
def getMap(id_2_diag_dir, id_2_order_dir):
        f = open(id_2_diag_dir,"r")
        id_treat = eval(f.read())
        for i in range(len(id_treat)):
            glob_features.append(id_treat[i])
        f = open(id_2_order_dir,"r")
        id_order = eval(f.read())
        for i in range(len(id_order)):
            glob_features.append(id_order[i])

def run_prompt_result(task_type, epoch,num_class, get_prediction, get_loss, model, feature_embedder, features, linear_net, show_attention = False, save_dir = ".", prompt_model=None):
    logits = get_prediction(
                model, feature_embedder, features, linear_net, prompt_model, training=False)
    if len(logits.shape) == 1:
        logits = logits.unsqueeze(dim=0)
    if logits.shape[1] > 1:
        nn_softmax = nn.Softmax()
        probs = nn_softmax(logits)
    else:
        nn_sigmoid = nn.Sigmoid()
        probs = nn_sigmoid(logits)
    
    labels = features['label'].to(device).float()
    if logits.shape[1] > 1:
        labels = one_hot(labels, num_class)
    loss = get_loss(logits, labels, None)
    out_probs = probs.cpu().detach().numpy()
    out_labels = np.array(labels.cpu())

    # if task_type != 'prompt_train':
    
    if task_type!='prompt_train':
        if logits.shape[1] > 1:
            out_probs = np.argmax(out_probs,axis=1)
            out_probs = np.eye(num_class)[out_probs].astype(int)
            return out_probs,out_labels, loss
        else:
            precision, recall, thresholds = precision_recall_curve(out_labels, out_probs)
            auc_precision_recall = auc(recall, precision)
            roc_auc = roc_auc_score(out_labels, out_probs)
            
            print('epoch %d loss %.5f, epoch AUC: %.5f ,AUCPR: %.5f' % (
                epoch + 1, loss, roc_auc, auc_precision_recall), flush=True)
    if task_type=='prompt_train':
        return loss

def calculateDecode(predicted_labels, true_labels):
    # 计算准确率
    predicted_labels = torch.stack(predicted_labels).tolist()
    true_labels = torch.stack(true_labels).tolist()
    accuracy = accuracy_score(true_labels, predicted_labels)

    # 计算精确度
    precision = precision_score(true_labels, predicted_labels, average='weighted')

    # 计算召回率
    recall = recall_score(true_labels, predicted_labels, average='weighted')

    # 计算F1分数
    f1 = f1_score(true_labels, predicted_labels, average='weighted')

    # 计算多类别混淆矩阵
    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    print("准确率:", accuracy)
    print("精确度:", precision)
    print("召回率:", recall)
    print("F1分数:", f1)
    print("多类别混淆矩阵:\n", conf_matrix)

def pretrian(task_type, epoch,num_class, dataloader, get_prediction, get_loss, optimizer, model, feature_embedder, linear_net, max_epoch = 200, show_attention = False, save_dir='.', prompt_model=None, decoder=None, decoder_optimizer=None):
    out_labels = []
    out_probs = []
    loss = 0
    decoded_loss = 0
    all_feature_num = 0
    predict_sum = 0
    for (idx, features) in enumerate(dataloader):
        if task_type == "train":
            loss, decoded_loss, predicted_labels, true_labels = run_pretrain_result(task_type, epoch,num_class, get_prediction, get_loss, model, feature_embedder, features, linear_net, decoder=decoder, decoder_optimizer=decoder_optimizer)
            if decoder != None:
                calculateDecode(predicted_labels, true_labels)
        elif task_type == "test" or task_type == "valid":
            with torch.no_grad():
                result, predict, batch_num, loss_epoch = run_pretrain_result(task_type, epoch,num_class, get_prediction, get_loss, model, feature_embedder, features, linear_net, decoder=decoder, decoder_optimizer=decoder_optimizer)
                loss += loss_epoch * batch_num
                all_feature_num += batch_num
                predict_sum += predict
        elif task_type=="prompt_train":
            loss = run_prompt_result(task_type, epoch,num_class, get_prediction, get_loss, model, feature_embedder, features, linear_net, show_attention, save_dir=save_dir, prompt_model=prompt_model)
        elif task_type == "prompt_valid" or task_type == "prompt_test":
            with torch.no_grad():
                out_probs_epoch,out_labels_epoch,loss_epoch = run_prompt_result(task_type, epoch,num_class, get_prediction, get_loss, model, feature_embedder, features, linear_net, show_attention, save_dir=save_dir, prompt_model=prompt_model)
                out_labels.append(out_labels_epoch)
                out_probs.append(out_probs_epoch)
                loss += loss_epoch * out_probs_epoch.shape[0]
        if task_type == 'train' or task_type == 'prompt_train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if decoder != None:
                decoder_optimizer.zero_grad()
                decoded_loss.backward()
                decoder_optimizer.step()
    if task_type == "test" or task_type == "valid":
        print('Task ======== ' + task_type)
        loss /= all_feature_num
        print("Loss = " + str(loss.item()))
        print("accuracy = " + str(predict_sum / all_feature_num))
        for i in range(len(result)):
            print(result[i])

    if task_type == "prompt_valid" or task_type == "prompt_test":
        print('Task ======== ' + task_type)
        out_labels = np.vstack(out_labels)
        out_probs = np.vstack(out_probs)
        loss /= out_labels.shape[0]
        flscore_micro = f1_score(out_labels, out_probs, average='micro')
        flscore_macro = f1_score(out_labels, out_probs, average='macro')
        flscore_weighted = f1_score(out_labels, out_probs, average='weighted')
        flscore_samples = f1_score(out_labels, out_probs, average='samples')
        for col in range(num_class):
            prob = out_probs[:,col]
            true_label = out_labels[:, col]
            f1 = f1_score(true_label, prob)
            print('==================' + str(col) + ' column ==============')
            print(f1)
        print('epoch %d loss %.5f, flscore_micro: %.5f, flscore_macro: %.5f, flscore_weighted: %.5f, flscore_samples: %.5f' % (
            epoch + 1, loss, flscore_micro, flscore_macro, flscore_weighted, flscore_samples), flush=True)


def init_embedding_with_bert(id_2_order_dir, id_2_diag_dir, fine_tune_bert=False, epoches=1):

    tokenizer = AutoTokenizer.from_pretrained("nghuyong/ernie-health-zh")
    model = SimcseModel(pretrained_model="nghuyong/ernie-health-zh", pooling="cls", dropout=0.2).to(device)
    model.to(device) 

    if fine_tune_bert == True:
        # model = SimcseModel(pretrained_model="nghuyong/ernie-health-zh", pooling="cls", dropout=0.2).to(device)
        model.train()
        feature_list = []
        f = open(id_2_order_dir,"r")
        id_treat = eval(f.read())
        for i in range(len(id_treat)):
            line = id_treat[0]
            line = line.strip()
            feature = tokenizer([line,line], max_length=64, truncation=True, padding='max_length', return_tensors='pt')
            feature_list.append(feature)
        
        f = open(id_2_diag_dir,"r")
        id_treat = eval(f.read())
        for i in range(len(id_treat)):
            line = id_treat[0]
            line = line.strip()
            feature = tokenizer([line,line], max_length=64, truncation=True, padding='max_length', return_tensors='pt')
            feature_list.append(feature)
        train_dataset = TrainDataset(feature_list, tokenizer, max_len=32)
        train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True,
                                      num_workers=1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
        for epoch in range(epoches):
            for batch_idx, data in enumerate(tqdm(train_dataloader)):
                step = epoch * len(train_dataloader) + batch_idx
                # [batch, n, seq_len] -> [batch * n, sql_len]
                sql_len = data['input_ids'].shape[-1]
                input_ids = data['input_ids'].view(-1, sql_len).to(device)
                attention_mask = data['attention_mask'].view(-1, sql_len).to(device)
                token_type_ids = data['token_type_ids'].view(-1, sql_len).to(device)

                out = model(input_ids, attention_mask, token_type_ids)
                loss = simcse_unsup_loss(out, device)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    model.eval()
    treat_list = []
    dx_list = []
    count = 0
    f = open(id_2_order_dir,"r")
    id_treat = eval(f.read())
    for i in range(len(id_treat)):
        line = id_treat[i]
        line = line.strip()
        feature = tokenizer(line, max_length=64, truncation=True, padding='max_length', return_tensors='pt')
        with torch.no_grad():
            sentence_embedding = model(feature['input_ids'].to(device), feature['attention_mask'].to(device), feature['token_type_ids'].to(device)).squeeze()
        treat_list.append(sentence_embedding)
    
    f1 = open(id_2_diag_dir,"r")
    id_dx = eval(f1.read())
    for i in range(len(id_dx)):
        line = id_dx[i]
        line = line.strip()
        feature = tokenizer(line, max_length=64, truncation=True, padding='max_length', return_tensors='pt')
        with torch.no_grad():
            sentence_embedding = model(feature['input_ids'].to(device), feature['attention_mask'].to(device), feature['token_type_ids'].to(device)).squeeze()
        dx_list.append(sentence_embedding)
    treat_list = torch.stack(treat_list).to(device)
    dx_list = torch.stack(dx_list).to(device)
    return dx_list, treat_list
def strs_2_list(strs):
    for i in range(len(strs)):
        strs[i] = eval(strs[i])
    return strs
def list_2_float_tensor(lists):
    for i in range(len(lists)):
        lists[i] = torch.FloatTensor(lists[i]).to(device)
    return lists
def list_2_int_tensor(lists):
    for i in range(len(lists)):
        lists[i] = torch.tensor(lists[i]).to(device)
    return lists
def freeze(layer):
    for param in layer.parameters():
        param.requires_grad = False


class GCTDataset(Dataset):
    def __init__(self, load_path, label, max_num_codes, prior_scalar, vocab_sizes, prompt_size, use_inf_mask, use_attention, encode='gbk', task_type='not_pretrain'):
        # self._label = label
        if use_attention:
            load_path = load_path[:-4] + '_' + label + '_attention' + load_path[-4:]
            self._attention = torch.load(load_path[:-4] + '.pth')
        self._data = pd.read_csv(load_path, encoding=encode)
        if use_attention:
            self._attention_idx = self._data['attention'].to_list()
        self._patientId = self._data['patientId'].to_list()
        self._dx_ids = self._data['dx_ids'].to_list()
        self._dx_ints = strs_2_list(self._data['dx_ints'].to_list())
        self._dx_ints_len = [len(self._dx_ints[i]) for i in range(len(self._dx_ints))]
        self._proc_ids = self._data['proc_ids'].to_list()
        self._proc_ints = strs_2_list(self._data['proc_ints'].to_list())
        self._proc_ints_len = [len(self._proc_ints[i]) for i in range(len(self._proc_ints))]
        self._prior_indices = list_2_float_tensor(strs_2_list(self._data['prior_indices'].to_list()))
        self._prior_values = list_2_float_tensor(strs_2_list(self._data['prior_values'].to_list()))
        self._dx_mask = torch.ones([len(self._dx_ints), max_num_codes['dx']]).to(device) #max_num_codes
        self._proc_mask = torch.ones([len(self._proc_ints), max_num_codes['px']]).to(device)  # max_num_codes
        self._prior_guide = []
        self._guide = []
        visit_guide = torch.FloatTensor(
            [prior_scalar] * max_num_codes['dx'] + [0.0] * max_num_codes['px'] * 1).to(device)
        mask_visit = torch.ones(1).to(device)

        self._dx_num = []
        self._proc_num = []
        self._label=[]
        self._task_type = task_type
        self._vocab_sizes = vocab_sizes
        # if task_type=='pretrain':
        #     for i in range(len(self._dx_ints)):
        #         r = random.randint(0, 1)
        #         tmp = {}
        #         if r == 0: 
        #             mask_idx = random.randint(0, len(self._dx_ints[i]) - 1)
        #             tmp['idx'] = self._dx_ints[i][mask_idx]
        #             self._dx_ints[i][mask_idx] = vocab_sizes['dx_ints']
        #             self._dx_mask[i][mask_idx] = 0
        #             tmp['type'] = 'dx_ints'
        #         else:
        #             mask_idx = random.randint(0, len(self._proc_ints[i]) - 1)
        #             tmp['idx'] = self._proc_ints[i][mask_idx]
        #             self._proc_ints[i][mask_idx] = vocab_sizes['proc_ints']
        #             self._proc_mask[i][mask_idx] = 0
        #             tmp['type'] = 'proc_ints'
        #         self._label.append(tmp)
        for i in range(len(self._dx_ints)):
            self._dx_num.append(len(self._dx_ints[i]))
            for j in range(len(self._dx_ints[i]), max_num_codes['dx']):
                self._dx_ints[i].append(vocab_sizes['dx_ints'])
                self._dx_mask[i][j] = 0
            self._dx_ints[i] = torch.tensor(self._dx_ints[i]).to(device)
        for i in range(len(self._proc_ints)):
            self._proc_num.append(len(self._proc_ints[i]))
            for j in range(len(self._proc_ints[i]), max_num_codes['px']):
                self._proc_ints[i].append(vocab_sizes['proc_ints'])
                self._proc_mask[i][j] = 0
            self._proc_ints[i] = torch.tensor(self._proc_ints[i]).to(device)
        row0 = torch.cat([
            torch.zeros([1, 1]).to(device),
            torch.ones([1, max_num_codes['dx']]).to(device),
            torch.zeros([1, max_num_codes['px']]).to(device)
        ],
            axis=1)

        row1 = torch.cat([
            torch.zeros([max_num_codes['dx'], 1 + max_num_codes['dx']]).to(device),
            torch.ones([max_num_codes['dx'], max_num_codes['px']]).to(device)
        ],
            axis=1)

        row2 = torch.zeros([max_num_codes['px'], 1 + max_num_codes['dx'] + max_num_codes['px']]).to(device)

        guide_init = torch.cat([row0, row1, row2], axis=0)
        for i in range(len(self._prior_indices)):
            if not use_attention:
                prior_idx = torch.reshape(self._prior_indices[i], [-1, 2])
                temp_idx = (prior_idx[:, 0] * 1000 + prior_idx[:, 1])
                sorted_idx = temp_idx.argsort()
                prior_idx = prior_idx[sorted_idx]
                prior_idx_shape = [max_num_codes['dx'] + max_num_codes['px'], max_num_codes['dx'] + max_num_codes['px']]
                sparse_prior = torch.sparse.FloatTensor(
                   prior_idx.long().T, self._prior_values[i], torch.Size(prior_idx_shape)).to(device)
                prior_guide = sparse_prior.to_dense()
                prior_guide = torch.cat(
                [visit_guide[None, :].repeat([1, 1]), prior_guide],
                axis=0)
                visit_guide_2 = torch.cat([torch.FloatTensor([0.0]).to(device), visit_guide], axis=0)
                prior_guide = torch.cat(
                    [visit_guide_2[:, None].repeat([1, 1]), prior_guide],
                    axis=1)
                mask = torch.cat([mask_visit, self._dx_mask[i], self._proc_mask[i]])
                prior_guide = (
                        prior_guide * mask[:, None] * mask[None, :] +
                        prior_scalar * torch.eye(1 + max_num_codes['dx'] + max_num_codes['px']).to(device)[:, :])
                degrees = torch.sum(prior_guide, dim=1)
                prior_guide = prior_guide / degrees[:, None]
                self._prior_guide.append(prior_guide)
            else:
                self._prior_guide.append(self._attention[self._attention_idx[i]][0])
            if use_inf_mask:
                mask = torch.cat([mask_visit, self._dx_mask[i], self._proc_mask[i]])
                guide = guide_init + guide_init.T
                guide = guide[:, :].repeat(1, 1)
                guide = (
                        guide * mask[:, None] * mask[None, :] +
                        torch.eye(1 + max_num_codes['dx'] + max_num_codes['px']).to(device)[:, :])
                self._guide.append(guide)
            else:
                guide = torch.ones(size=guide_init.shape).to(device)
                self._guide.append(guide)

        self._label = self._data[label].to_list()

    def __getitem__(self, index):
        with torch.no_grad():
            result_dic = {"patientId": self._patientId[index],
                        "dx_ids": self._dx_ids[index],
                        "dx_ints": self._dx_ints[index],
                        "proc_ids": self._proc_ids[index],
                        "proc_ints": self._proc_ints[index],
                        "guide": self._guide[index],
                        "prior_guide": self._prior_guide[index],
                        "dx_num": self._dx_num[index],
                        "proc_num": self._proc_num[index],
                        "mask": {
                            "dx_ints": self._dx_mask[index],
                            "proc_ints": self._proc_mask[index]},
                        "label": self._label[index]}
            if self._task_type=='pretrain' or self._task_type=='mask_predict':
                r = random.randint(0, 1)
                tmp = {}
                if r == 0: 
                    mask_idx = random.randint(0, self._dx_ints_len[index] - 1)
                    tmp['idx'] = self._dx_ints[index][mask_idx].item()
                    tmp_dx_ints = self._dx_ints[index].clone()
                    tmp_dx_ints[mask_idx] = self._vocab_sizes['dx_ints']
                    result_dic['dx_ints'] = tmp_dx_ints
                    # self._dx_mask[index][mask_idx] = 0
                    tmp_dx_mask = self._dx_mask[index].clone()
                    tmp_dx_mask[mask_idx] = 0
                    result_dic['mask']['dx_ints'] = tmp_dx_mask
                    tmp['type'] = 'dx_ints'
                else:
                    mask_idx = random.randint(0, self._proc_ints_len[index] - 1)
                    tmp['idx'] = self._proc_ints[index][mask_idx].item()
                    # self._proc_ints[index][mask_idx] = self._vocab_sizes['proc_ints'] - 1
                    tmp_proc_ints = self._proc_ints[index].clone()
                    tmp_proc_ints[mask_idx] = self._vocab_sizes['proc_ints']
                    result_dic['proc_ints'] = tmp_proc_ints
                    # self._proc_mask[index][mask_idx] = 0
                    tmp_proc_mask = self._proc_mask[index].clone()
                    tmp_proc_mask[mask_idx] = 0
                    result_dic['mask']['proc_ints'] = tmp_proc_mask
                    tmp['type'] = 'proc_ints'
                self._label[index] = tmp
                result_dic['label'] = self._label[index]  # for the case of the first time this method is called.

            
            return result_dic

    def __len__(self):
        return self._data.shape[0]
class MLPDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPDecoder, self).__init__()
        
        # 定义MLP的层
        self.fc1 = nn.Linear(input_dim, hidden_dim).to(device)
        self.relu = nn.ReLU().to(device)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim).to(device)
        self.fc3 = nn.Linear(hidden_dim, output_dim).to(device)
        self.softmax = nn.Softmax(dim=1).to(device)
    def forward(self, x, true_labels):

        # 前向传播
        # x = self.fc1(x)
        # x = self.relu(x)
        # x = self.fc2(x)
        # x = self.relu(x)
        x = self.fc3(x)
        decoded_res = self.softmax(x)
        crossLoss = nn.CrossEntropyLoss().to(device)
        loss = crossLoss(decoded_res.float().to(device), true_labels.to(device))
        predict_decode = torch.argmax(decoded_res, dim=1)
        true_lable = torch.argmax(true_labels, dim=1)
        return loss, predict_decode, true_lable
class FeatureKeyEmbedding(nn.Module):
    def __init__(self, vocab_sizes, feature_keys, embedding_size, use_bert = False, use_position = False,id_2_order_dir = '.', id_2_diag_dir = '.', fine_tune_bert = False, bert_epoches=1):
        super(FeatureKeyEmbedding, self).__init__()
        self._params = nn.ModuleList()
        self._use_bert = use_bert
        if self._use_bert:
            dx_emb, treat_emb = init_embedding_with_bert(id_2_order_dir, id_2_diag_dir, fine_tune_bert, bert_epoches)
            emb = nn.Embedding(vocab_sizes['dx_ints'] + 1, 768, padding_idx=vocab_sizes['dx_ints']).to(device)
            dx_emb = torch.cat((dx_emb, emb.weight[vocab_sizes['dx_ints']:]))
            emb.weight = torch.nn.Parameter(dx_emb)
            self._params.append(emb)

            emb = nn.Embedding(vocab_sizes['proc_ints'] + 1, 768, padding_idx=vocab_sizes['proc_ints']).to(device)
            treat_emb = torch.cat((treat_emb, emb.weight[vocab_sizes['proc_ints']:]))
            emb.weight = torch.nn.Parameter(treat_emb)
            self._params.append(emb)
            
        else:
            for feature_key in feature_keys:
                vocab_size = vocab_sizes[feature_key]
                emb = nn.Embedding(vocab_size + 1, embedding_size,  padding_idx=vocab_size)
                # torch.nn.init.kaiming_normal_(emb, nonlinearity='leaky_relu')
                self._params.append(emb)#aplly dummy embedding
        self._position = PositionalEncoding(embedding_size, 0.1, use_position)
        self._params.append(nn.Embedding(1, embedding_size))
        self._params.append(nn.Embedding(1, embedding_size))
        # self._decoder = MLPDecoder(embedding_size, embedding_size, vocab_sizes['dx_ints'] + vocab_sizes['proc_ints'])

    def forward(self, features, str, predict_pos=None, batch_size=1):
        if str == 'predict':
            tmp = []
            for i in range(len(predict_pos)):
                tmp.append((predict_pos[i] == 'proc_ints') + 1)
            tmp = torch.tensor(tmp).to(device)
            x = self._position(self._params[3](features).reshape(1, self._params[3](features).shape[0]).repeat(batch_size, 1)[:,None,:], tmp)
            return x
        elif str == 'visit':
            x = self._position(self._params[2](features), 0)
            return x
        elif str == 'dx_ints':
            x = self._position(self._params[0](features), 1)
            return x
        elif str == 'proc_ints':
            x = self._position(self._params[1](features), 2)
            return x
    # def decode(self, features, true_labels):
    #     decoded_res = self._decoder(features)
    #     crossLoss = nn.CrossEntropyLoss().to(device)
    #     loss = crossLoss(decoded_res.float().to(device), true_labels.to(device))
    #     return loss


class LinearNet(nn.Module):
    def __init__(self, input_size, out_size):
        super(LinearNet, self).__init__()
        self._net = nn.Linear(input_size, out_size)

    def forward(self, features):
        x = self._net(features)
        return x
class MaskLM(nn.Module):
    def __init__(self,vocab_sizes, feature_keys,num_inputs=768, **kwargs):
        super(MaskLM,self).__init__()
        self._vocab_sizes =  vocab_sizes
        self._vocab_size = 0
        for key in feature_keys:
            self._vocab_size = self._vocab_sizes[key] + self._vocab_size + 1 # NONE SENSE
        self._vocab_size = self._vocab_size + 2
        self.dense = nn.Sequential(nn.Linear(num_inputs,num_inputs),
                                 nn.ReLU(),
                                 nn.LayerNorm(num_inputs, eps=1e-12))
        self.decoder = nn.Linear(num_inputs, self._vocab_size, bias=False)
        self.decoder.to(device)
        self.dense.to(device)
    
    def get_vocab_size(self):
        return self._vocab_size
    
    def forward(self,X, encode_weight):
        # weight = torch.cat([encode_weight[:self._vocab_size['dx_ints'],], encode_weight[self._vocab_size['dx_ints']:self._vocab_size:self._vocab_size['dx_ints'] + self._vocab_size['proc_ints'],]])
        self.decoder.weight = nn.Parameter(encode_weight)
        self.decoder.bias = nn.Parameter(torch.zeros(self._vocab_size))
        self.decoder.to(device)
        return self.decoder(self.dense(X))
    

class PromptEmbedding(torch.nn.Module):
    def __init__(self, prompt_num, embedding_size, dropout_p=0.5):
        super(PromptEmbedding, self).__init__()
        self._prompt_num = prompt_num  
        self._embedding = nn.Embedding(prompt_num + 1, embedding_size,  padding_idx=prompt_num).to(device)
    
    def getPromptNum(self):
        return self._prompt_num
    
    def tokenView(self, feature_map, key):
        feature = feature_map[key]
        batch_size = feature.shape[0]
        return self._embedding(torch.tensor([i for i in range(self._prompt_num)]).to(device)).repeat([batch_size, 1, 1]),torch.ones(batch_size, self._prompt_num).to(device)

class FeatureEmbedding(object):
    def __init__(self, vocab_sizes, feature_keys, embedding_size, use_bert = False,use_position=False, id_2_order_dir = '.', id_2_diag_dir = '.', fine_tune_bert = False, bert_epoches=1):
        # self._params = {}
        self._feature_keys = feature_keys
        self._vocab_sizes = vocab_sizes
        self._embedding_size = embedding_size

        self._embed_model = FeatureKeyEmbedding(self._vocab_sizes, self._feature_keys, self._embedding_size, use_bert,use_position, id_2_order_dir, id_2_diag_dir, fine_tune_bert, bert_epoches).to(device)

    def getPredict(self, feature_map):
        embeddings = {}
        masks = {}
        for key in self._feature_keys:
          feature = feature_map[key]
        batch_size = feature.shape[0]
        embeddings = self._embed_model(torch.tensor(0).to(device), 'predict', feature_map['label']['type'], batch_size)
        masks = torch.ones(batch_size).to(device)[:, None]
        return embeddings, masks
    
    def getEncodeWeight(self):
        return torch.cat([self._embed_model._params[i].weight for i in range(len(self._embed_model._params))])
    
    def lookupLabels(self, labels):
        res = []
        for i in range(len(labels['type'])):
            curNum = 0
            for key in self._feature_keys:
                if key != labels['type'][i]:
                    curNum = curNum + self._vocab_sizes[key]
                else:
                    curNum = curNum + labels['idx'][i]
                    res.append(curNum)
                    break
        return torch.stack(res)

    def get_vocab_and_key(self):
        return {"vocab":self._vocab_sizes,"feature_key":self._feature_keys}

    def lookup(self, feature_map, max_num_codes):
        """Look-up function.

        This function converts the SparseTensor of integers to a dense Tensor of
        tf.float32.

        Args:
          feature_map: A dictionary of SparseTensors for each feature.
          max_num_codes: The maximum number of how many feature there can be inside
            a single visit, per feature. For example, if this is set to 50, then we
            are assuming there can be up to 50 diagnosis codes, 50 treatment codes,
            and 50 lab codes. This will be used for creating the prior matrix.

        Returns:
          embeddings: A dictionary of dense representation Tensors for each feature.
          masks: A dictionary of dense float32 Tensors for each feature, that will
            be used as a mask in the downstream tasks.
        """
        masks = {}
        embeddings = {}
        for key in self._feature_keys:
          feature = feature_map[key]
          mask = feature_map["mask"][key]
          embeddings[key] = self._embed_model(feature, key)
          masks[key] = mask
        batch_size = feature.shape[0]
        embeddings['visit'] = self._embed_model(torch.tensor(0).to(device), 'visit').reshape(1, self._embedding_size)[None, :, :].repeat(batch_size, 1, 1)

        masks['visit'] = torch.ones(batch_size).to(device)[:, None]

        return embeddings, masks

class MLPPredictor(nn.Module):
    def __init__(self,
                 embedding_size=128,
                 dropout_rate=0.25):
        super(MLPPredictor, self).__init__()
        self._embedding_size = embedding_size
        self._dropout_rate = dropout_rate
        self._mlp_deep = nn.ModuleList()
        self._mlp_modules = nn.ModuleList()

        for i in range(7):
            self._mlp_deep.append(nn.Sequential(nn.Linear(embedding_size, embedding_size), nn.ReLU()))
        self._mlp_deep.append(nn.Linear(embedding_size, embedding_size))

        self._mlp_modules.append(nn.Sequential(nn.Linear(128, 64), nn.ReLU()))
        self._mlp_modules.append(nn.Sequential(nn.Linear(64, 32), nn.ReLU()))
        self._mlp_modules.append(nn.Sequential(nn.Linear(32, 16), nn.ReLU()))
        self._mlp_modules.append(nn.Sequential(nn.Linear(16, 8), nn.ReLU()))
        self._mlp_modules.append(nn.Sequential(nn.Linear(8, 4), nn.ReLU()))
        self._mlp_modules.append(nn.Sequential(nn.Linear(4, 2), nn.ReLU()))
        self._mlp_modules.append(nn.Sequential(nn.Linear(2, 1)))

    def forward(self, features, training):
        for layer in self._mlp_deep:
            features = layer(features)
            if training:
                drop_layer = nn.Dropout(p=self._dropout_rate)
                features = drop_layer(features)
        features = features.sum(dim=1)
        for layer in self._mlp_modules:
            features = layer(features)
            if training:
                drop_layer = nn.Dropout(p=self._dropout_rate)
                features = drop_layer(features)
        return features


class GraphConvolutionalTransformer(nn.Module):
    """Graph Convolutional Transformer class.

    This is an implementation of Graph Convolutional Transformer. With a proper
    set of options, it can be used as a vanilla Transformer.
    """

    def __init__(self,
               embedding_size=128,
               num_transformer_stack=3,
               num_feedforward=2,
               num_attention_heads=1,
               ffn_dropout=0.1,
               attention_normalizer='softmax',
               multihead_attention_aggregation='concat',
               directed_attention=False,
               use_inf_mask=True,
               use_prior=True,
               **kwargs):
        """Init function.

        Args:
        embedding_size: The size of the dimension for hidden layers.
        num_transformer_stack: The number of Transformer blocks.
        num_feedforward: The number of layers in the feedforward part of
            Transformer.
        num_attention_heads: The number of attention heads.
        ffn_dropout: Dropout rate used inside the feedforward part.
        attention_normalizer: Use either 'softmax' or 'sigmoid' to normalize the
        attention values.
        multihead_attention_aggregation: Use either 'concat' or 'sum' to handle
            the outputs from multiple attention heads.
        directed_attention: Decide whether you want to use the unidirectional
        attention, where information accumulates inside the dummy visit node.
        use_inf_mask: Decide whether you want to use the guide matrix. Currently
            unused.
        use_prior: Decide whether you want to use the conditional probablility
            information. Currently unused.
        **kwargs: Other arguments to tf.keras.layers.Layer init.
        """

        super(GraphConvolutionalTransformer, self).__init__()
        self._hidden_size = embedding_size
        self._num_stack = num_transformer_stack
        self._num_feedforward = num_feedforward
        self._num_heads = num_attention_heads
        self._ffn_dropout = ffn_dropout
        self._attention_normalizer = attention_normalizer
        self._multihead_aggregation = multihead_attention_aggregation
        self._directed_attention = directed_attention
        self._use_inf_mask = use_inf_mask
        self._use_prior = use_prior

        self._layers_Q = nn.ModuleList()
        self._layers_K = nn.ModuleList()
        self._layers_V = nn.ModuleList()
        self._layers_ffn = nn.ModuleList()
        self._layers_head_agg = nn.ModuleList()

        for i in range(self._num_stack):
          self._layers_Q.append(
              nn.Linear(
                  self._hidden_size, self._hidden_size * self._num_heads, bias=False))
          self._layers_K.append(
              nn.Linear(
                  self._hidden_size, self._hidden_size * self._num_heads, bias=False))
          self._layers_V.append(
              nn.Linear(
                  self._hidden_size, self._hidden_size * self._num_heads, bias=False))

          if self._multihead_aggregation == 'concat':
            self._layers_head_agg.append(
              nn.Linear(
                  self._hidden_size, self._hidden_size, bias=False))

          self._layers_ffn.append(nn.ModuleList())
          # Don't need relu for the last feedforward.
          for _ in range(self._num_feedforward - 1):
            self._layers_ffn[i].append(
              nn.Sequential(nn.Linear(
                  self._hidden_size, self._hidden_size), nn.ReLU()))
          self._layers_ffn[i].append(
              nn.Linear(
                  self._hidden_size, self._hidden_size))

    def feedforward(self, features, stack_index, training=None):
        """Feedforward component of Transformer.

        Args:
        features: 3D float Tensor of size (batch_size, num_features,
            embedding_size). This is the input embedding to GCT.
        stack_index: An integer to indicate which Transformer block we are in.
        training: Whether to run in training or eval mode.

        Returns:
        Latent representations derived from this feedforward network.
        """
        for i in range(self._num_feedforward):
            features = self._layers_ffn[stack_index][i](features)
            if training:
                layer = nn.Dropout(p=self._ffn_dropout)
                features = layer(features)
        return features

    def qk_op(self,
            features,
            stack_index,
            batch_size,
            num_codes,
            attention_mask,
            inf_mask=None,
            directed_mask=None):
        """Attention generation part of Transformer.

        Args:
        features: 3D float Tensor of size (batch_size, num_features,
        embedding_size). This is the input embedding to GCT.
         stack_index: An integer to indicate which Transformer block we are in.
        batch_size: The size of the mini batch.
        num_codes: The number of features (i.e. codes) given as input.
        attention_mask: A Tensor for suppressing the attention on the padded
            tokens.
        inf_mask: The guide matrix to suppress the attention values to zeros for
            certain parts of the attention matrix (e.g. diagnosis codes cannot
            attend to other diagnosis codes).
        directed_mask: If the user wants to only use the upper-triangle of the
            attention for uni-directional attention flow, we use this strictly lower
            triangular matrix filled with infinity.

        Returns:
        The attention distribution derived from the QK operation.
        """

        q = self._layers_Q[stack_index](features)
        q = torch.reshape(q,
                    [batch_size, num_codes, self._hidden_size, self._num_heads])

        k = self._layers_K[stack_index](features)
        k = torch.reshape(k,
                   [batch_size, num_codes, self._hidden_size, self._num_heads])

        # Need to transpose q and k to (2, 0, 1)
        q = q.permute(0, 3, 1, 2)
        k = k.permute(0, 3, 2, 1)

        pre_softmax = torch.matmul(q, k) / torch.sqrt(torch.tensor(float(self._hidden_size)))

        pre_softmax -= attention_mask[:, None, None, :]

        if inf_mask is not None:
            pre_softmax -= inf_mask[:, None, :, :]

        if directed_mask is not None:
            pre_softmax -= directed_mask

        if self._attention_normalizer == 'softmax':
            m1 = nn.Softmax(dim=3)
            attention = m1(pre_softmax)
        else:
            m1 = nn.Sigmoid(pre_softmax)
            attention = m1(pre_softmax)
        return attention

    def forward(self, features, masks, guide=None, prior_guide=None, training=None):
        """This function transforms the input embeddings.

        This function converts the SparseTensor of integers to a dense Tensor of
        tf.float32.

        Args:
        features: 3D float Tensor of size (batch_size, num_features,
            embedding_size). This is the input embedding to GCT.
        masks: 3D float Tensor of size (batch_size, num_features, 1). This holds
            binary values to indicate which parts are padded and which are not.
        guide: 3D float Tensor of size (batch_size, num_features, num_features).
            This is the guide matrix.
        prior_guide: 3D float Tensor of size (batch_size, num_features,
            num_features). This is the conditional probability matrix.
        training: Whether to run in training or eval mode.

        Returns:
        features: The final layer of GCT.
        attentions: List of attention values from all layers of GCT. This will be
            used later to regularize the self-attention process.
        """

        batch_size = features.shape[0]
        num_codes = features.shape[1]

        # Use the given masks to create a negative infinity Tensor to suppress the
        # attention weights of the padded tokens. Note that the given masks has
        # the shape (batch_size, num_codes, 1), so we remove the last dimension
        # during the process.
        mask_clone = masks[:, :, 0]
        max_matrix = torch.ones(mask_clone.shape).to(device)
        max_matrix = max_matrix.fill_(3.4028234e+38).float()
        zero_matrix = torch.zeros(mask_clone.shape).to(device)
        attention_mask = torch.where(mask_clone != 0, zero_matrix, max_matrix)

        inf_mask = None
        if self._use_inf_mask:
            max_matrix = torch.ones(guide.shape).to(device).float()
            max_matrix = max_matrix.fill_(3.4028234e+38)
            zero_matrix = torch.zeros(guide.shape).to(device)
            inf_mask = torch.where(guide != 0, zero_matrix, max_matrix)

        directed_mask = None
        # if self._directed_attention:
        #   inf_matrix = tf.fill([num_codes, num_codes], tf.float32.max)
        #   inf_matrix = tf.matrix_set_diag(inf_matrix, tf.zeros(num_codes))
        #   directed_mask = tf.matrix_band_part(inf_matrix, -1, 0)[None, None, :, :]

        attention = None
        attentions = []
        for i in range(self._num_stack):
          features = masks * features

          if self._use_prior and i == 0:
            attention = prior_guide[:, None, :, :].repeat([1, 1, 1, 1])
          else:
            attention = self.qk_op(features, i, batch_size, num_codes,
                                   attention_mask, inf_mask, directed_mask)

          attentions.append(attention)

          v = self._layers_V[i](features)
          v = v.reshape([batch_size, num_codes, self._hidden_size, self._num_heads])
          v = v.permute(0, 3, 1, 2)
          # post_attention is (batch, num_heads, num_codes, hidden_size)
          post_attention = torch.matmul(attention, v)

          if self._num_heads == 1:
            post_attention = torch.squeeze(post_attention, axis=1)
          elif self._multihead_aggregation == 'concat':
            # post_attention is (batch, num_codes, num_heads, hidden_size)
            post_attention = post_attention.permute(0, 2, 1, 3)
            # post_attention is (batch, num_codes, num_heads*hidden_size)
            post_attention = post_attention.reshape(batch_size, num_codes, -1)
            # post attention is (batch, num_codes, hidden_size)
            post_attention = self._layers_head_agg[i](post_attention)
          else:
            post_attention = torch.sum(post_attention, dim=1)

          # Residual connection + layer normalization
          post_attention += features
          post_attention = nn.LayerNorm([self._hidden_size], elementwise_affine=False)(post_attention)#may error wrong

          # Feedforward component + residual connection + layer normalization
          post_ffn = self.feedforward(post_attention, i, training)
          post_ffn += post_attention
          post_ffn = nn.LayerNorm([self._hidden_size], elementwise_affine=False)(post_ffn)

          features = post_ffn

        return features * masks, attentions
def create_matrix_vdp(features, mask, use_prior, use_inf_mask, max_num_codes,
                      prior_scalar):
      guide = None
      if use_inf_mask:
          guide = features['guide']
      prior_guide = None
      if use_prior:
          prior_guide = features['prior_guide']

      return guide, prior_guide

class EHRTransformer(object):
    """Transformer-based EHR encounter modeling algorithm.
    All features within each encounter are put through multiple steps of
    self-attention. There is a dummy visit embedding in addition to other
    feature embeddings, which can be used for encounter-level predictions.
    """

    def __init__(self,
               gct_params,
               task_type='fine_tune',
               feature_keys=['dx_ints', 'proc_ints'],
               label_key='label.readmission',
               vocab_sizes={'dx_ints':5000, 'proc_ints':10000},
               prompt_size=10,
               feature_set='vdp',
               max_num_codes={'dx':31,'px':339},
               prior_scalar=0.5,
               reg_coef=0.1,
               num_classes=1,
               use_bert=False,
               use_position=False,
               fine_tune_bert=False,
               bert_epoches=1,
               learning_rate=1e-3,
               batch_size=32,
               id_2_diag_dir='.',
               id_2_order_dir='.',
               encode='gbk',
               use_attention=False, #default not use attention
               epoches = 1000000,
               input_path=None):
        """Init function.

        Args:
          gct_params: A dictionary parameteres to be used inside GCT class. See GCT
            comments for more information.
          feature_keys: A list of feature names you want to use. (e.g. ['dx_ints,
            'proc_ints', 'lab_ints'])
          vocab_sizes: A dictionary of vocabularize sizes for each feature. (e.g.
            {'dx_ints': 1001, 'proc_ints': 1001, 'lab_ints': 1001})
          feature_set: Use 'vdpl' to indicate your features are diagnosis codes,
            treatment codes, and lab codes. Use 'vdp' to indicate your features are
            diagnosis codes and treatment codes.
          max_num_codes: The maximum number of how many feature there can be inside
            a single visit, per feature. For example, if this is set to 50, then we
            are assuming there can be up to 50 diagnosis codes, 50 treatment codes,
            and 50 lab codes. This will be used for creating the prior matrix.
          prior_scalar: A float value between 0.0 and 1.0 to be used to hard-code
            the diagnoal elements of the prior matrix.
          reg_coef: A coefficient to decide the KL regularization balance when
            training GCT.
          num_classes: This is set to 1, because this implementation only supports
            graph-level binary classification.
          learning_rate: Learning rate for Adam optimizer.
          batch_size: Batch size.
        """
        seed_it(1314)
        self._feature_keys = feature_keys
        self._label_key = label_key
        self._vocab_sizes = vocab_sizes
        self._prompt_size = prompt_size
        self._task_type = task_type
        if task_type != 'pretrain' and task_type != 'prompt' and task_type != 'mask_predict':
            self._prompt_size = 0
        self._feature_set = feature_set
        self._max_num_codes = max_num_codes
        self._prior_scalar = prior_scalar
        self._reg_coef = reg_coef
        self._num_classes = num_classes
        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self._use_bert = use_bert
        self._use_position = use_position
        self._fine_tune_bert = fine_tune_bert
        self._is_training = gct_params['training']
        self._id_2_diag_dir = id_2_diag_dir
        self._id_2_order_dir = id_2_order_dir
        self._encode=encode

        self._gct_params = gct_params
        self._embedding_size = gct_params['embedding_size']
        self._num_transformer_stack = gct_params['num_transformer_stack']
        self._use_inf_mask = gct_params['use_inf_mask']
        self._use_prior = gct_params['use_prior']
        self._input_path = input_path
        self._epoches = epoches
        self._use_attention = use_attention
        self._bert_epoches = bert_epoches

        getMap(id_2_diag_dir,id_2_order_dir)

        self._max_num_codes[list(self._max_num_codes.keys())[-1]] = self._max_num_codes[list(self._max_num_codes.keys())[-1]] + prompt_size
        self._train_data_set = GCTDataset(self._input_path + '/train.csv', self._label_key, self._max_num_codes, self._prior_scalar, self._vocab_sizes, self._prompt_size, self._use_inf_mask, use_attention, encode=self._encode,task_type=self._task_type)
        self._valid_data_set = GCTDataset(self._input_path + '/validation.csv', self._label_key, self._max_num_codes, self._prior_scalar, self._vocab_sizes, self._prompt_size, self._use_inf_mask, use_attention, encode=self._encode, task_type=self._task_type)
        self._test_data_set = GCTDataset(self._input_path + '/test.csv', self._label_key, self._max_num_codes, self._prior_scalar, self._vocab_sizes, self._prompt_size, self._use_inf_mask, use_attention, encode=self._encode, task_type=self._task_type)
        self._train_data_loader = DataLoader(self._train_data_set, batch_size=self._batch_size, shuffle=True, num_workers=0)
        self._valid_data_loader = DataLoader(self._valid_data_set, batch_size=int(self._valid_data_set.__len__()/5), shuffle=False, num_workers=0)
        self._test_data_loader = DataLoader(self._test_data_set, batch_size=int(self._test_data_set.__len__()/5), shuffle=False, num_workers=0)


    

    def get_mask_prediction(self, model, feature_embedder, features, linear_net, mask_diagnoses, batch_tensor, training=False):
        """Accepts features and produces logits and attention values.

                    Args:
                      features: A dictionary of SparseTensors for each sequence feature.
                      training: A boolean value to indicate whether the predictions are for
                        training or inference. If set to True, dropouts will take effect.

                    Returns:
                      logits: Logits for prediction.
                      attentions: List of attention values from all layers of GCT. Pass this to
                        get_loss to regularize the attention generation mechanism.
                    """
        # 1. Embedding lookup

        embedding_dict, mask_dict = feature_embedder.lookup(
            features, self._max_num_codes)

        mask_diagnoses_embeddings = feature_embedder._embed_model(mask_diagnoses, 'dx_ints')
        # 2. Concatenate embeddings and masks into a single tensor.
        keys = ['visit'] + self._feature_keys
        embeddings = torch.cat([embedding_dict[key] for key in keys], axis=1)
        masks = torch.cat([mask_dict[key] for key in keys], axis=1)
        masks[batch_tensor, mask_diagnoses + 1] = 0
        if self._feature_set == 'vdp':
            guide, prior_guide = create_matrix_vdp(features, masks, self._use_prior,
                                                   self._use_inf_mask,
                                                   self._max_num_codes,
                                                   self._prior_scalar)
        else:
            sys.exit(0)

        # 3. Process embeddings with GCT
        # s = time.time()
        hidden, attentions = model(
            embeddings, masks[:, :, None], guide, prior_guide, training)
        # e = time.time()
        # print((e-s)*1000.0)

        # 4. Generate logits
        predict_embeddings = hidden[:, 0, :]



        return predict_embeddings, mask_diagnoses_embeddings, attentions
    
    def prompt_prediction(self, model, feature_embedder, features, linear_net, prompt_model, training=True, model_type='gct'):
        for i in range(len(self._feature_keys)):
            if i == len(self._feature_keys) - 1:
                features[self._feature_keys[i]] = features[self._feature_keys[i]][:,:-self._prompt_size]
                features['mask'][self._feature_keys[i]] = features['mask'][self._feature_keys[i]][:,:-self._prompt_size]
        embedding_dict, mask_dict = feature_embedder.lookup(
            features, self._max_num_codes)
        pretrain_embedding, pretrain_mask = prompt_model.tokenView(features, self._feature_keys[0])
        embedding_dict['predict'] = pretrain_embedding
        mask_dict['predict'] = pretrain_mask
        # 2. Concatenate embeddings and masks into a single tensor.
        keys = ['visit', 'predict'] + self._feature_keys
        embeddings = torch.cat([embedding_dict[key] for key in keys], axis=1)
        masks = torch.cat([mask_dict[key] for key in keys], axis=1)

        if model_type == 'gct':
            if self._feature_set == 'vdp':
                guide, prior_guide = create_matrix_vdp(features, masks, self._use_prior,
                                                       self._use_inf_mask,
                                                       self._max_num_codes,
                                                       self._prior_scalar)
            else:
                sys.exit(0)

            # 3. Process embeddings with GCT
            # s = time.time()
            hidden, attentions = model(
                embeddings, masks[:, :, None], guide, prior_guide, training)
            # e = time.time()
            # print((e-s)*1000.0)

            # 4. Generate logits
            pre_logit = hidden[:, 0, :]
            pre_logit = pre_logit.reshape(-1, self._embedding_size)
            logits = linear_net(pre_logit)
            logits = torch.squeeze(logits)

        elif model_type == 'mlp':
            logits = model(embeddings, training)
            logits = torch.squeeze(logits)
            attentions = None
        return logits


    def pretrain_prediction(self, model, feature_embedder, features, linear_net,training=True, model_type='gct'):
        for i in range(len(self._feature_keys)):
            if i == len(self._feature_keys) - 1:
                features[self._feature_keys[i]] = features[self._feature_keys[i]][:,:-1]
                features['mask'][self._feature_keys[i]] = features['mask'][self._feature_keys[i]][:,:-1]
        embedding_dict, mask_dict = feature_embedder.lookup(
            features, self._max_num_codes)
        pretrain_embedding, pretrain_mask = feature_embedder.getPredict(
            features)
        embedding_dict['predict'] = pretrain_embedding
        mask_dict['predict'] = pretrain_mask
        # 2. Concatenate embeddings and masks into a single tensor.
        keys = ['visit', 'predict'] + self._feature_keys
        embeddings = torch.cat([embedding_dict[key] for key in keys], axis=1)
        masks = torch.cat([mask_dict[key] for key in keys], axis=1)

        if model_type == 'gct':
            if self._feature_set == 'vdp':
                guide, prior_guide = create_matrix_vdp(features, masks, self._use_prior,
                                                       self._use_inf_mask,
                                                       self._max_num_codes,
                                                       self._prior_scalar)
            else:
                sys.exit(0)

            # 3. Process embeddings with GCT
            # s = time.time()
            hidden, attentions = model(
                embeddings, masks[:, :, None], guide, prior_guide, training)
            # e = time.time()
            # print((e-s)*1000.0)

            # 4. Generate logits
            pre_logit = hidden[:, 1, :]
            
        return linear_net(pre_logit, feature_embedder.getEncodeWeight()), hidden
    
    def get_prediction(self, model, feature_embedder, features, linear_net, training=False, model_type='gct'):
        """Accepts features and produces logits and attention values.

                    Args:
                      features: A dictionary of SparseTensors for each sequence feature.
                      training: A boolean value to indicate whether the predictions are for
                        training or inference. If set to True, dropouts will take effect.

                    Returns:
                      logits: Logits for prediction.
                      attentions: List of attention values from all layers of GCT. Pass this to
                        get_loss to regularize the attention generation mechanism.
                    """
        # 1. Embedding lookup

        embedding_dict, mask_dict = feature_embedder.lookup(
            features, self._max_num_codes)

        # 2. Concatenate embeddings and masks into a single tensor.
        keys = ['visit'] + self._feature_keys
        embeddings = torch.cat([embedding_dict[key] for key in keys], axis=1)
        masks = torch.cat([mask_dict[key] for key in keys], axis=1)

        if model_type == 'gct':
            if self._feature_set == 'vdp':
                guide, prior_guide = create_matrix_vdp(features, masks, self._use_prior,
                                                       self._use_inf_mask,
                                                       self._max_num_codes,
                                                       self._prior_scalar)
            else:
                sys.exit(0)

            # 3. Process embeddings with GCT
            # s = time.time()
            hidden, attentions = model(
                embeddings, masks[:, :, None], guide, prior_guide, training)
            # e = time.time()
            # print((e-s)*1000.0)

            # 4. Generate logits
            pre_logit = hidden[:, 0, :]
            pre_logit = pre_logit.reshape(-1, self._embedding_size)
            logits = linear_net(pre_logit)
            logits = torch.squeeze(logits)

        elif model_type == 'mlp':
            logits = model(embeddings, training)
            logits = torch.squeeze(logits)
            attentions = None
        return logits, attentions, hidden

    def get_mask_diagnose_loss(self, predict_embeddings, mask_embeddings, attentions):

        log_p = torch.log(predict_embeddings + 1e-12)
        log_q = torch.log(mask_embeddings + 1e-12)
        kl_term = predict_embeddings * (log_p - log_q)
        kl_term = torch.sum(kl_term, dim=-1)
        loss = torch.mean(kl_term)

        if self._use_prior:
            kl_terms = []
            attention_tensor = torch.stack(attentions)
            for i in range(1, self._num_transformer_stack):
                log_p = torch.log(attention_tensor[i - 1] + 1e-12)
                log_q = torch.log(attention_tensor[i] + 1e-12)
                kl_term = attention_tensor[i - 1] * (log_p - log_q)
                kl_term = torch.sum(kl_term, dim=-1)
                kl_term = torch.mean(kl_term)
                kl_terms.append(kl_term)

            reg_term = torch.mean(torch.stack(kl_terms))
            loss += self._reg_coef * reg_term

    def get_loss(self, logits, labels, attentions):
        """Creates a loss tensor.

        Args:
          logits: Logits for prediction. This is obtained by calling get_prediction.
          labels: Labels for prediction.
          attentions: List of attention values from all layers of GCT. This is
            obtained by calling get_prediction.

        Returns:
          Loss tensor. If we use the conditional probability matrix, then GCT's
          attention mechanism will be regularized using KL divergence.
        """

        if logits.shape[1] > 1: 
            softmax_logits = nn.Softmax()(logits)
            labels = labels.float()
            loss = nn.BCELoss()(softmax_logits, labels)


        else:
            sigmoid_logits = nn.Sigmoid()(logits)
            loss = nn.BCELoss()(sigmoid_logits, labels)

        if self._use_prior:
            kl_terms = []
            attention_tensor = torch.stack(attentions)
            for i in range(1, self._num_transformer_stack):
                log_p = torch.log(attention_tensor[i - 1] + 1e-12)
                log_q = torch.log(attention_tensor[i] + 1e-12)
                kl_term = attention_tensor[i - 1] * (log_p - log_q)
                kl_term = torch.sum(kl_term, dim=-1)
                kl_term = torch.mean(kl_term)
                kl_terms.append(kl_term)

            reg_term = torch.mean(torch.stack(kl_terms))
            loss += self._reg_coef * reg_term

        return loss

    # def updateAttention(self, model_path):
    #     feature_embedder = FeatureEmbedding(self._vocab_sizes, self._feature_keys, self._embedding_size)
    #     linear_net = LinearNet(self._embedding_size, self._num_classes)
    #     model = GraphConvolutionalTransformer(**self._gct_params)
    #     linear_net = linear_net.to(device)
    #     model = model.to(device)

    #     state_dict = torch.load(model_path + 'model_name.pth')
    #     linear_net.load_state_dict(state_dict['linear_net'])
    #     model.load_state_dict(state_dict['model'])
    #     feature_embedder._embed_model.load_state_dict(state_dict['feature_embedder._embed_model'])

    #     linear_net.eval()
    #     feature_embedder._embed_model.eval()
    #     model.eval()
    #     self._train_data_loader = DataLoader(self._train_data_set, batch_size=self._train_data_set.__len__(), shuffle=True,
    #                                          num_workers=0)
    #     loader_list = {"train": self._train_data_loader,
    #                    "validation": self._valid_data_loader,
    #                    "test": self._test_data_loader}
    #     keys = ["train", "validation", "test"]
    #     # 替换矩阵
    #     for key in keys:
    #         for (idx, features) in enumerate(loader_list[key]):
    #             with torch.no_grad():
    #                 logits, attentions = self.get_prediction(
    #                     model, feature_embedder, features, linear_net, training=self._gct_params["training"])
    #                 dataFrame = pd.read_csv(self._input_path + '/' + key +'.csv')
    #                 dataFrame['attention'] = [i for i in range(len(attentions[2]))]
    #                 all_list = []
    #                 for i in range(len(features['patientId'])):
    #                     index = dataFrame[dataFrame.patientId == features['patientId'][i]].index.to_list()[0]
    #                     all_list.append(index)
    #                     dataFrame.loc[index, 'attention'] = i
    #                 torch.save(attentions[2], self._input_path + '/' + key + '_' + self._label_key + '_attention.pth')
    #                 dataFrame.to_csv(self._input_path + '/' + key + '_' + self._label_key +'_attention.csv')
    def prompt_train(self, model_path):
        feature_embedder = FeatureEmbedding(self._vocab_sizes, self._feature_keys, self._embedding_size, self._use_bert,self._use_position, self._id_2_order_dir, self._id_2_diag_dir, self._fine_tune_bert, self._bert_epoches)
        linear_net = LinearNet(self._embedding_size, self._num_classes)
        model = GraphConvolutionalTransformer(**self._gct_params)
        linear_net = linear_net.to(device)
        model = model.to(device)
        vocab_size = 0
        for key in self._feature_keys:
            vocab_size = self._vocab_sizes[key] + vocab_size # NONE SENSE
        decoder = MLPDecoder(self._embedding_size, self._embedding_size, vocab_size)
        decoder_optimizer = optim.Adam([{"params": decoder.parameters()}],lr=self._learning_rate*10, betas=(0.9, 0.999), eps=1e-8)

        state_dict = torch.load(model_path + '/pretrain_and_fine_tune_model.pth')
        # linear_net.load_state_dict(state_dict['linear_net'])
        model.load_state_dict(state_dict['model'])
        feature_embedder._embed_model.load_state_dict(state_dict['feature_embedder._embed_model'])
        prompt_nodes = PromptEmbedding(self._prompt_size, self._embedding_size)
        linear_net.train()
        prompt_nodes.train()
        feature_embedder._embed_model.eval()
        model.eval()
        optimizer = optim.Adam([{"params": linear_net.parameters()},
                    {"params": prompt_nodes.parameters()}],lr=self._learning_rate, betas=(0.9, 0.999), eps=1e-8)
        result_epoch = self._epoches
        starttime = time.time()
        for epoch in range(result_epoch):
            if epoch >= 0:# eval
                endtime = time.time()
                print('train %d epoch time %.5f' % (
                    epoch + 1, float(endtime - starttime) * 1000.0), flush=True)
                linear_net.eval()
                prompt_nodes.eval()
                pretrian(task_type = "prompt_valid", epoch = epoch, num_class = self._num_classes,dataloader = self._valid_data_loader, get_prediction=self.prompt_prediction, get_loss=self.get_loss, optimizer=optimizer, model=model, feature_embedder=feature_embedder, linear_net=linear_net, prompt_model=prompt_nodes)
                pretrian(task_type = "prompt_test", epoch = epoch, num_class = self._num_classes, dataloader = self._test_data_loader,get_prediction=self.prompt_prediction, get_loss=self.get_loss, optimizer=optimizer, model=model, feature_embedder=feature_embedder, linear_net=linear_net, prompt_model=prompt_nodes)
            linear_net.train()
            prompt_nodes.train()
            pretrian(task_type = "prompt_train", epoch = epoch, num_class = self._num_classes, dataloader = self._train_data_loader,get_prediction=self.prompt_prediction, get_loss=self.get_loss, optimizer=optimizer, model=model, feature_embedder=feature_embedder, linear_net=linear_net, max_epoch = result_epoch, prompt_model=prompt_nodes)

    def fine_tune(self, model_path):
        feature_embedder = FeatureEmbedding(self._vocab_sizes, self._feature_keys, self._embedding_size, self._use_bert,self._use_position, self._id_2_order_dir, self._id_2_diag_dir,self._fine_tune_bert, self._bert_epoches)
        linear_net = LinearNet(self._embedding_size, self._num_classes)
        vocab_size = 0
        for key in self._feature_keys:
            vocab_size = self._vocab_sizes[key] + vocab_size # NONE SENSE
        decoder = MLPDecoder(self._embedding_size, self._embedding_size, vocab_size)
        decoder_optimizer = optim.Adam([{"params": decoder.parameters()}],lr=self._learning_rate*10, betas=(0.9, 0.999), eps=1e-8)
        model = GraphConvolutionalTransformer(**self._gct_params)
        linear_net = linear_net.to(device)
        model = model.to(device)

        state_dict = torch.load(model_path + '/pretrain_model.pth')
        # linear_net.load_state_dict(state_dict['linear_net'])
        model.load_state_dict(state_dict['model'])
        feature_embedder._embed_model.load_state_dict(state_dict['feature_embedder._embed_model'])
        linear_net.train()
        feature_embedder._embed_model.train()
        model.train()
        optimizer = optim.Adam([{"params": linear_net.parameters()},
                    {"params": model.parameters()},
                    {"params": feature_embedder._embed_model.parameters()}],lr=self._learning_rate, betas=(0.9, 0.999), eps=1e-8)
        result_epoch = self._epoches
        starttime = time.time()
        for epoch in range(result_epoch):
            if epoch >= 0:# eval
                endtime = time.time()
                linear_net.eval()
                feature_embedder._embed_model.eval()
                model.eval()
                print('train %d epoch time %.5f' % (
                    epoch + 1, float(endtime - starttime) * 1000.0), flush=True)
                run_model(task_type = "valid", epoch = epoch, num_class = self._num_classes,dataloader = self._valid_data_loader, get_prediction=self.get_prediction, get_loss=self.get_loss, optimizer=optimizer, model=model, feature_embedder=feature_embedder, linear_net=linear_net, decoder=decoder,decoder_optimizer=decoder_optimizer)

                run_model(task_type = "test", epoch = epoch, num_class = self._num_classes, dataloader = self._test_data_loader,get_prediction=self.get_prediction, get_loss=self.get_loss, optimizer=optimizer, model=model, feature_embedder=feature_embedder, linear_net=linear_net, decoder=decoder,decoder_optimizer=decoder_optimizer)
            linear_net.train()
            feature_embedder._embed_model.train()
            model.train()
            run_model(task_type = "train", epoch = epoch, num_class = self._num_classes, dataloader = self._train_data_loader,get_prediction=self.get_prediction, get_loss=self.get_loss, optimizer=optimizer, model=model, feature_embedder=feature_embedder, linear_net=linear_net, max_epoch = result_epoch, decoder=decoder,decoder_optimizer=decoder_optimizer)
            torch.save({'model': model.state_dict(), "linear_net": linear_net.state_dict(),
                            "feature_embedder._embed_model": feature_embedder._embed_model.state_dict()},
                           model_path + '/pretrain_and_fine_tune_model.pth')

    def real_pretrain(self, model_path, use_bert=False):
        feature_embedder = FeatureEmbedding(self._vocab_sizes, self._feature_keys, self._embedding_size, self._use_bert,self._use_position, self._id_2_order_dir, self._id_2_diag_dir,self._fine_tune_bert, self._bert_epoches)
        vocab_size = 0
        for key in self._feature_keys:
            vocab_size = self._vocab_sizes[key] + vocab_size # NONE SENSE
        decoder = MLPDecoder(self._embedding_size, self._embedding_size, vocab_size)
        decoder_optimizer = optim.Adam([{"params": decoder.parameters()}],lr=self._learning_rate*10, betas=(0.9, 0.999), eps=1e-8)
        linear_net = MaskLM(self._vocab_sizes,self._feature_keys, self._embedding_size)
        model = GraphConvolutionalTransformer(**self._gct_params)
        linear_net = linear_net.to(device)
        model = model.to(device)

        optimizer = optim.Adam([{"params": linear_net.parameters()},
                    {"params": model.parameters()},
                    {"params": feature_embedder._embed_model.parameters()}],lr=self._learning_rate, betas=(0.9, 0.999), eps=1e-8)
        linear_net.train()
        feature_embedder._embed_model.train()
        model.train()
        starttime = time.time()
        result_epoch = self._epoches
        for epoch in range(result_epoch):
            linear_net.train()
            feature_embedder._embed_model.train()
            model.train()
            pretrian(task_type = "train", epoch = epoch, num_class = self._num_classes, dataloader = self._train_data_loader,get_prediction=self.pretrain_prediction, get_loss=self.get_loss, optimizer=optimizer, model=model, feature_embedder=feature_embedder, linear_net=linear_net, max_epoch = result_epoch, decoder=decoder, decoder_optimizer=decoder_optimizer)
            linear_net.eval()
            feature_embedder._embed_model.eval()
            model.eval()
            pretrian(task_type = "train", epoch = epoch, num_class = self._num_classes, dataloader = self._test_data_loader,get_prediction=self.pretrain_prediction, get_loss=self.get_loss, optimizer=optimizer, model=model, feature_embedder=feature_embedder, linear_net=linear_net, max_epoch = result_epoch, decoder=decoder, decoder_optimizer=decoder_optimizer)
            pretrian(task_type = "train", epoch = epoch, num_class = self._num_classes, dataloader = self._valid_data_loader,get_prediction=self.pretrain_prediction, get_loss=self.get_loss, optimizer=optimizer, model=model, feature_embedder=feature_embedder, linear_net=linear_net, max_epoch = result_epoch, decoder=decoder, decoder_optimizer=decoder_optimizer)
        torch.save({'model': model.state_dict(), "linear_net": linear_net.state_dict(),
                            "feature_embedder._embed_model": feature_embedder._embed_model.state_dict()},
                           model_path + '/pretrain_model.pth')
        return
    
    def pretrain_train(self, model_path, use_bert=False):
        feature_embedder = FeatureEmbedding(self._vocab_sizes, self._feature_keys, self._embedding_size, self._use_bert,self._use_position, self._id_2_order_dir, self._id_2_diag_dir,self._fine_tune_bert, self._bert_epoches)
        vocab_size = 0
        for key in self._feature_keys:
            vocab_size = self._vocab_sizes[key] + vocab_size # NONE SENSE
        decoder = MLPDecoder(self._embedding_size, self._embedding_size, vocab_size)
        decoder_optimizer = optim.Adam([{"params": decoder.parameters()}],lr=self._learning_rate*10, betas=(0.9, 0.999), eps=1e-8)
        linear_net = MaskLM(self._vocab_sizes,self._feature_keys, self._embedding_size)
        model = GraphConvolutionalTransformer(**self._gct_params)
        linear_net = linear_net.to(device)
        model = model.to(device)

        optimizer = optim.Adam([{"params": linear_net.parameters()},
                    {"params": model.parameters()},
                    {"params": feature_embedder._embed_model.parameters()}],lr=self._learning_rate, betas=(0.9, 0.999), eps=1e-8)
        linear_net.train()
        feature_embedder._embed_model.train()
        model.train()
        starttime = time.time()
        result_epoch = self._epoches
        for epoch in range(result_epoch):
            linear_net.train()
            feature_embedder._embed_model.train()
            model.train()
            pretrian(task_type = "train", epoch = epoch, num_class = self._num_classes, dataloader = self._train_data_loader,get_prediction=self.pretrain_prediction, get_loss=self.get_loss, optimizer=optimizer, model=model, feature_embedder=feature_embedder, linear_net=linear_net, max_epoch = result_epoch, decoder=decoder, decoder_optimizer=decoder_optimizer)
            linear_net.eval()
            feature_embedder._embed_model.eval()
            model.eval()
            pretrian(task_type = "test", epoch = epoch, num_class = self._num_classes, dataloader = self._test_data_loader,get_prediction=self.pretrain_prediction, get_loss=self.get_loss, optimizer=optimizer, model=model, feature_embedder=feature_embedder, linear_net=linear_net, max_epoch = result_epoch, decoder=decoder, decoder_optimizer=decoder_optimizer)
            pretrian(task_type = "valid", epoch = epoch, num_class = self._num_classes, dataloader = self._valid_data_loader,get_prediction=self.pretrain_prediction, get_loss=self.get_loss, optimizer=optimizer, model=model, feature_embedder=feature_embedder, linear_net=linear_net, max_epoch = result_epoch, decoder=decoder, decoder_optimizer=decoder_optimizer)
        torch.save({'model': model.state_dict(), "linear_net": linear_net.state_dict(),
                            "feature_embedder._embed_model": feature_embedder._embed_model.state_dict()},
                           model_path + '/pretrain_model.pth')
        return
        
        


    def test(self, model_path, use_bert=False):
        feature_embedder = FeatureEmbedding(self._vocab_sizes, self._feature_keys, self._embedding_size, self._use_bert,self._use_position, self._id_2_order_dir, self._id_2_diag_dir,self._fine_tune_bert, self._bert_epoches)
        linear_net = LinearNet(self._embedding_size, self._num_classes)
        model = GraphConvolutionalTransformer(**self._gct_params)
        linear_net = linear_net.to(device)
        model = model.to(device)

        if self._use_attention:
            state_dict = torch.load(model_path + 'attention_model_name.pth')
        else:
            state_dict = torch.load(model_path + 'model_name.pth')
        linear_net.load_state_dict(state_dict['linear_net'])
        model.load_state_dict(state_dict['model'])
        feature_embedder._embed_model.load_state_dict(state_dict['feature_embedder._embed_model'])


        linear_net.eval()
        feature_embedder._embed_model.eval()
        model.eval()
        run_model(task_type = "valid", epoch = 0, num_class = self._num_classes,dataloader = self._valid_data_loader, get_prediction=self.get_prediction, get_loss=self.get_loss, optimizer=None, model=model, feature_embedder=feature_embedder, linear_net=linear_net)
        run_model(task_type = "test", epoch = 0, num_class = self._num_classes, dataloader = self._test_data_loader,get_prediction=self.get_prediction, get_loss=self.get_loss, optimizer=None, model=model, feature_embedder=feature_embedder, linear_net=linear_net)
    
    def run_gct_model(self, model_path):
        if self._is_training:
            self.train_and_pred(model_dir = model_path, model_type='gct', use_bert = self._use_bert)
        else:
            self.test(model_path = model_path, use_bert=self._use_bert)

    def train_mlp(self, model_dir, model_type):
        feature_embedder = FeatureEmbedding(self._vocab_sizes, self._feature_keys, self._embedding_size, self._use_bert,self._use_position, self._id_2_order_dir, self._id_2_diag_dir,self._fine_tune_bert, self._bert_epoches)
        model = MLPPredictor().to(device)
        optimizer = optim.Adam([{"params": model.parameters()},
                                {"params": feature_embedder._embed_model.parameters()}], lr=self._learning_rate,
                               betas=(0.9, 0.999), eps=1e-8)
        feature_embedder._embed_model.train()
        model.train()
        starttime = time.time()
        result_epoch = self._epoches
        for epoch in range(result_epoch):
            if epoch % 100 == 0 or epoch == result_epoch - 1:  # eval
                endtime = time.time()
                print('train %d epoch time %.5f' % (
                    epoch + 1, float(endtime - starttime) * 1000.0), flush=True)
                torch.save({'model': model.state_dict(), "feature_embedder._embed_model": feature_embedder._embed_model.state_dict()},
                               model_dir + '/mlp_model.pth')
                model.eval()
                feature_embedder._embed_model.eval()
                for (idx, features) in enumerate(self._valid_data_loader):
                    with torch.no_grad():
                        logits, attentions = self.get_prediction(
                            model, feature_embedder, features, linear_net=None, training=False, model_type=model_type)
                        nn_sigmoid = nn.Sigmoid()
                        probs = nn_sigmoid(logits)
                        predictions = {
                            'probabilities': probs,
                            'logits': logits,
                        }
                        labels = features['label'].to(device).float()
                        loss = self.get_loss(logits, labels, attentions)

                        out_probs = probs.cpu().detach().numpy()
                        out_labels = np.array(labels.cpu())
                        precision, recall, thresholds = precision_recall_curve(out_labels, out_probs)
                        auc_precision_recall = auc(recall, precision)
                        roc_auc = roc_auc_score(out_labels, out_probs)
                        print('valid %d loss %.5f, epoch AUC: %.5f ,AUCPR: %.5f' % (
                            epoch + 1, loss, roc_auc, auc_precision_recall), flush=True)

                for (idx, features) in enumerate(self._test_data_loader):
                    with torch.no_grad():
                        logits, attentions = self.get_prediction(
                            model, feature_embedder, features, linear_net=None, training=False, model_type=model_type)
                        nn_sigmoid = nn.Sigmoid()
                        probs = nn_sigmoid(logits)
                        predictions = {
                            'probabilities': probs,
                            'logits': logits,
                        }
                        labels = features['label'].to(device).float()
                        loss = self.get_loss(logits, labels, attentions)

                        out_probs = probs.cpu().detach().numpy()
                        out_labels = np.array(labels.cpu())
                        precision, recall, thresholds = precision_recall_curve(out_labels, out_probs)
                        auc_precision_recall = auc(recall, precision)
                        roc_auc = roc_auc_score(out_labels, out_probs)
                        print('test %d loss %.5f, epoch AUC: %.5f ,AUCPR: %.5f' % (
                            epoch + 1, loss, roc_auc, auc_precision_recall), flush=True)

                feature_embedder._embed_model.train()
                model.train()
            for (idx, features) in enumerate(self._train_data_loader):
                # s = time.time()
                logits, attentions = self.get_prediction(
                    model, feature_embedder, features, linear_net=None, training=self._gct_params["training"],
                    model_type=model_type)

                nn_sigmoid = nn.Sigmoid()
                probs = nn_sigmoid(logits)
                labels = features['label'].to(device).float()
                loss = self.get_loss(logits, labels, attentions)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def run_gct_pretrain(self, model_path, task_type):
        if task_type == 'pretrain':
            self.real_pretrain(model_path = model_path, use_bert = self._use_bert)
        elif task_type == 'prompt':
            self.prompt_train(model_path = model_path)
        elif task_type == 'fine_tune':
            self.fine_tune(model_path=model_path)
        elif task_type == 'mask_predict':
            self.pretrain_train(model_path = model_path, use_bert = self._use_bert)
    def train_and_pred(self, model_dir, model_type='gct', use_bert=False):
        # data preprocess

        feature_embedder = FeatureEmbedding(self._vocab_sizes, self._feature_keys, self._embedding_size, self._use_bert,self._use_position, self._id_2_order_dir, self._id_2_diag_dir,self._fine_tune_bert, self._bert_epoches)
        linear_net = LinearNet(self._embedding_size, self._num_classes)
        if model_type == 'gct':
            model = GraphConvolutionalTransformer(**self._gct_params)
        elif model_type == 'mlp':
            model = MLPPredictor()
        linear_net = linear_net.to(device)
        model = model.to(device)

        optimizer = optim.Adam([{"params": linear_net.parameters()},
                    {"params": model.parameters()},
                    {"params": feature_embedder._embed_model.parameters()}],lr=self._learning_rate, betas=(0.9, 0.999), eps=1e-8)
        linear_net.train()
        feature_embedder._embed_model.train()
        model.train()
        starttime = time.time()
        result_epoch = self._epoches
        for epoch in range(result_epoch):
            if epoch >= 0:# eval
                endtime = time.time()
                print('train %d epoch time %.5f' % (
                    epoch + 1, float(endtime - starttime) * 1000.0), flush=True)
                if self._use_attention:
                    torch.save({'model': model.state_dict(), "linear_net": linear_net.state_dict(),
                                "feature_embedder._embed_model": feature_embedder._embed_model.state_dict()},
                               model_dir + '/attention_model_name.pth')
                else:
                    torch.save({'model': model.state_dict(), "linear_net": linear_net.state_dict(),
                            "feature_embedder._embed_model": feature_embedder._embed_model.state_dict()},
                           model_dir + '/model_name.pth')
                linear_net.eval()
                feature_embedder._embed_model.eval()
                model.eval()
                run_model(task_type = "valid", epoch = epoch, num_class = self._num_classes,dataloader = self._valid_data_loader, get_prediction=self.get_prediction, get_loss=self.get_loss, optimizer=optimizer, model=model, feature_embedder=feature_embedder, linear_net=linear_net)

                run_model(task_type = "test", epoch = epoch, num_class = self._num_classes, dataloader = self._test_data_loader,get_prediction=self.get_prediction, get_loss=self.get_loss, optimizer=optimizer, model=model, feature_embedder=feature_embedder, linear_net=linear_net)
            linear_net.train()
            feature_embedder._embed_model.train()
            model.train()
            run_model(task_type = "train", epoch = epoch, num_class = self._num_classes, dataloader = self._train_data_loader,get_prediction=self.get_prediction, get_loss=self.get_loss, optimizer=optimizer, model=model, feature_embedder=feature_embedder, linear_net=linear_net, max_epoch = result_epoch)

        return

    def train_mask_diagnose(self, model_dir, model_type='mask_diagnose'):
        # data preprocess

        feature_embedder = FeatureEmbedding(self._vocab_sizes, self._feature_keys, self._embedding_size, self._use_bert,self._use_position, self._id_2_order_dir, self._id_2_diag_dir,self._fine_tune_bert, self._bert_epoches)
        linear_net = LinearNet(self._embedding_size, self._num_classes)
        model = GraphConvolutionalTransformer(**self._gct_params)
        linear_net = linear_net.to(device)
        model = model.to(device)

        optimizer = optim.Adam([{"params": linear_net.parameters()},
                    {"params": model.parameters()},
                    {"params": feature_embedder._embed_model.parameters()}],lr=self._learning_rate, betas=(0.9, 0.999), eps=1e-8)
        linear_net.train()
        feature_embedder._embed_model.train()
        model.train()
        starttime = time.time()
        batch_tensor = torch.arange(0, 32)
        result_epoch = self._epoches
        for epoch in range(result_epoch):
            if epoch % 100 == 0  and epoch != 0 or epoch == result_epoch - 1:# eval
                endtime = time.time()
                print('train %d epoch time %.5f' % (
                    epoch + 1, float(endtime - starttime) * 1000.0), flush=True)
                if self._use_attention:
                    torch.save({'model': model.state_dict(), "linear_net": linear_net.state_dict(),
                                "feature_embedder._embed_model": feature_embedder._embed_model.state_dict()},
                               model_dir + '/attention_model_name.pth')
                else:
                    torch.save({'model': model.state_dict(), "linear_net": linear_net.state_dict(),
                            "feature_embedder._embed_model": feature_embedder._embed_model.state_dict()},
                           model_dir + '/mask_diagnose_model.pth')
                linear_net.eval()
                feature_embedder._embed_model.eval()
                model.eval()
                for (idx, features) in enumerate(self._valid_data_loader):
                    with torch.no_grad():
                        logits, attentions = self.get_prediction(
                            model, feature_embedder, features, linear_net, training=False)
                        nn_sigmoid = nn.Sigmoid()
                        probs = nn_sigmoid(logits)
                        predictions = {
                            'probabilities': probs,
                            'logits': logits,
                        }
                        labels = features['label'].to(device).float()
                        loss = self.get_loss(logits, labels, attentions)

                        out_probs = probs.cpu().detach().numpy()
                        out_labels = np.array(labels.cpu())
                        precision, recall, thresholds = precision_recall_curve(out_labels, out_probs)
                        auc_precision_recall = auc(recall, precision)
                        roc_auc = roc_auc_score(out_labels, out_probs)
                        print('valid %d loss %.5f, epoch AUC: %.5f ,AUCPR: %.5f' % (
                            epoch + 1, loss, roc_auc, auc_precision_recall), flush=True)

                for (idx, features) in enumerate(self._test_data_loader):
                    with torch.no_grad():
                        logits, attentions = self.get_prediction(
                            model, feature_embedder, features, linear_net, training=False)
                        nn_sigmoid = nn.Sigmoid()
                        probs = nn_sigmoid(logits)
                        predictions = {
                            'probabilities': probs,
                            'logits': logits,
                        }
                        labels = features['label'].to(device).float()
                        loss = self.get_loss(logits, labels, attentions)

                        out_probs = probs.cpu().detach().numpy()
                        out_labels = np.array(labels.cpu())
                        precision, recall, thresholds = precision_recall_curve(out_labels, out_probs)
                        auc_precision_recall = auc(recall, precision)
                        roc_auc = roc_auc_score(out_labels, out_probs)
                        print('test %d loss %.5f, epoch AUC: %.5f ,AUCPR: %.5f' % (
                            epoch + 1, loss, roc_auc, auc_precision_recall), flush=True)

                linear_net.train()
                feature_embedder._embed_model.train()
                model.train()
            for (idx, features) in enumerate(self._train_data_loader):
                # s = time.time()
                # random.randint((0, features['dx_num'][i] for i in range(self._batch_size)))
                mask_diagnoses = []
                for i in range(self._batch_size):
                    mask_diagnoses.append(random.randint(0, features['dx_num'][i] - 1))
                mask_diagnoses = torch.tensor(mask_diagnoses).to(device)
                #mask diagnose
                features['dx_ints'][batch_tensor, mask_diagnoses] = self._vocab_sizes['dx_ints']
                features['prior_guide'][batch_tensor, :, mask_diagnoses] = 0
                features['prior_guide'][batch_tensor, mask_diagnoses, :] = 0
                predict_embeddings, mask_diagnoses_embeddings, attentions = self.get_mask_prediction(
                    model, feature_embedder, features, linear_net, mask_diagnoses, batch_tensor, training=self._gct_params["training"])
                loss = self.get_mask_diagnose_loss(predict_embeddings, mask_diagnoses_embeddings, attentions)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


        return