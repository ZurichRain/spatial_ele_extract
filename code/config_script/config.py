# 这个文件里写自己的所有的配置代码
'''
bert_base_2020 模型: submodel: qslink_model olink_model movelink_model  联合qslink 或者olink或者movelink抽取
bert_base 模型: submodel: softmax_model 仅仅是bert后添加一个softmax层
'''

import os
import torch

# train_CP_dir='/data2/fwang/baseline/data/spatial_data/Traning/CP/'
# model_name= 'bert_base_2020'
model_name= 'bert_base_2020'
sub_model_name = 'qslink_model'
model_parameter_adjustment_name=model_name+'_'+sub_model_name+'_lr_2e_5_bz_1'
model_parameter_adjustment_eval_result_name=model_parameter_adjustment_name+'_log'

bert_pathname = "/data2/fwang/transformers_model/bert-base-uncased/"
# T5small_pathname='/data2/fwang/transformers_model/T5_small/'
T5base_pathname='/data2/fwang/transformers_model/T5_base/'

train_CP_dir='/data2/fwang/baseline/data/spatial_data/all_train_data/'# 训练集55个文档
# train_CP_dir='/data2/fwang/baseline/data/spatial_data/all_train_data_59/'# 训练集59个文档
# train_CP_dir='/data2/fwang/baseline/data/spatial_data/one_e_train/'

vail_data_dir='/data2/fwang/baseline/data/spatial_data/spaceeval_trial_data/'
# vail_data_dir='/data2/fwang/baseline/data/spatial_data/all_val_data/'
# vail_data_dir='/data2/fwang/baseline/data/spatial_data/one_e_val/'

# test_data_dir='/data2/fwang/baseline/data/spatial_data/test_task8/Test.configuration3/CP/'
# test_data_dir='/data2/fwang/baseline/data/spatial_data/test_all_with_gold/'
test_data_dir='/data2/fwang/baseline/data/spatial_data/test_all_with_all_gold_tag/' #测试集中有14个文档
# test_data_dir='/data2/fwang/baseline/data/spatial_data/test_all_with_all_gold_tag_16/' #测试集中有16个文档 其中两篇来自训练集

train_log_dir='/data2/fwang/spatial_ele_extract/code/log/'+model_name+'_log/'+sub_model_name+'/train_log/'
test_log_dir='/data2/fwang/spatial_ele_extract/code/log/'+model_name+'_log/'+sub_model_name+'/test_log/'

save_train_model_dir='/data2/fwang/spatial_ele_extract/code/model/'+model_name+'_model/'+sub_model_name+'/'
save_train_model_file=save_train_model_dir+model_parameter_adjustment_name

save_train_result_dir='/data2/fwang/spatial_ele_extract/code/vail_test_data_result/'+model_name+'_model/'+sub_model_name+'/'

save_train_result_file=save_train_result_dir+model_parameter_adjustment_eval_result_name

# bert_name='bert-base-uncased'
bert_name='bert-base-uncased'

device='cuda:4' if torch.cuda.is_available() else 'cpu'

clip_grad = 5 #解决 梯度爆炸或者消失问题的参数
epoch_num = 100 
min_epoch_num = 3
learning_rate = 2e-5
load_before = False
batch_size=1
link_batch_size=1

patience = 0.0002# loss不下降的精度
patience_num = 10 #连续多少轮之后loss不下降就break

full_fine_tuning=True 

weight_decay=0.1 #梯度衰减

qsrole_labels=['qstj','qsld','qstr']
spatial_qsrole_label2id={
    'O':0,
    'B-qstj':1,
    'I-qstj':2,
    'B-qsld':3,
    'I-qsld':4,
    'B-qstr':5,
    'I-qstr':6
}
# 空间元素对应的标签
ele_labels=['spatial_entity','nonmotion_event','motion','spatial_signal','motion_signal','measure','place','path']
# 空间元素对应的标签
spatial_ele_label2id={
    'O':0,
    'B-spatial_entity':1,
    'I-spatial_entity':2,
    'B-nonmotion_event':3,
    'I-nonmotion_event':4,
    'B-motion':5,
    'I-motion':6,
    'B-spatial_signal':7,
    'I-spatial_signal':8,
    'B-motion_signal':9,
    'I-motion_signal':10,
    'B-measure':11,
    'I-measure':12,
    'B-place':13,
    'I-place':14,
    'B-path':15,
    'I-path':16
}