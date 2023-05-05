# 数据预处理
./preprocess/preprocess.py是数据预处理的入口文件，调用pre_diagnose.py、pre_order.py、merge.py三个文件。  
程序启动方式：python preprocess/preprocess.py config_path
config_path为配置路径  
例子：
python preprocess/preprocess.py config.json

# 模型训练
./train_morlity_main.py是模型的启动程序，从清洗好的数据中读取数据放入模型训练。  
程序启动方式：CUDA_VISIBLE_DEVICES={i} python train_morlity_main.py config_path  
i为显卡号，config_path为配置路径  
例子：
CUDA_VISIBLE_DEVICES=0 python train_morlity_main.py ./config.json


# 可设置参数
input_path: 源数据文件输入路径  
output_path: 处理之后数据的输出路径，也即模型的数据输入路径  
model_dir: 模型的保存路径  
diagnosis_name: diagnosis数据的文件名（不需要路径了，只需要写文件名）  
orders_name: orders数据的文件名（不需要路径了，只需要写文件名）  
encode: 数据文件的编码方式，默认为gbk，如果不行可以试一下utf-8  
model_data_encode: 处理之后输入给模型的数据编码，推荐utf-8  
score: 相似文本匹配的阈值，分数大于该值时则合并  
condtions: orders_class的筛选，输入应该是一个字符串组成的列表  
sample_num: 同一个visit中orders的抽样数，如果不需要抽样则为-1  
diag_max_num: 同一个visit最多的diagnose数量，请勿手动更改  
oder_max_num: 同一个visit最多的order数量，请勿手动更改  
diag_voclen: 同一个visit最多的diagnose数量，请勿手动更改  
order_voclen: 同一个visit最多的order数量，请勿手动更改  
use_bert：是否开启bert，如果开启还应确保系统中有bert，使用的bert版本是ernie-health-zh（模型下载：https://huggingface.co/nghuyong/ernie-health-zh） 
use_prior: 是否使用条件概率矩阵初始化,  
num_classes：预测类别数，单分类问题设置成2，多分类问题设置成n即可,  
training: 是否训练。如果置为false则不训练，只测试，即test模式，此时需要模型目录下有训练好的模型,  
label_key：标签名字，当前设置为label.expired，对应清洗数据之后的标签  
use_inf_mask: 是否使用mask矩阵将diagnose内部和orders内部关系mask,  

# 超参数（在train_morlity_main.py中自行配置）
"embedding_size": 词向量长度,  
"num_transformer_stack": transformer模型层数，控制模型大小,  
"num_feedforward": 全连接层数量,  
"num_attention_heads": 多头注意力机制中注意力头个数,  
"ffn_dropout": dropout参数,  
"attention_normalizer": 可选"softmax"或"sigmoid",  
"multihead_attention_aggregation": 暂时只支持"concat"，多头注意力结果的连接方式,  
reg_coef：当前模型训练结果是两个loss相加，调整reg_coef可以控制模型更关注于哪一个损失，loss = loss(下游任务) + reg_coef*loss(attention与条件概率矩阵的差距)  
learning_rate：学习率  
batch_size：一次抽样批次的大小  
epoches：数据迭代次数  

如果要查看模型验证集中训练的最好结果，需要先打印出日志，再使用get_best_result.py对日志分析。  
例如：
CUDA_VISIBLE_DEVICES=0 python train_morlity_main.py ./config.json > ./log.txt  
python get_best_result.py ./log.txt  

便可以打印出日志中验证集flscore_macro最高对应的epoch，以及该epoch对应的test和valid的一些结果。  

# 5.5更新
在config中加入了prompt_num字段和task_type字段（可选参数为"pretrain", "prompt", "fine_tune），定义prompt的节点数量，pretrain过程也需要定义该参数。  
新的pretrain, prompt, fine_tune方法调用train_pretrain_prompt_finetune.py即可，只需修改config的参数即可完成调用  
建议的组合有pretrain+prompt, pretrain+fine_tune。  
pretrain阶段只打印loss，不使用标签。  
pretrain采用的MLM算法，随机mask掉一些节点来预测。  





