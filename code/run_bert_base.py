'''
最基本的空间元素抽取 使用bert+softmax

'''
# import utils
import sys
import os

from sympy import im
sys.path.append('./code/')
import config_script.config as config
import logging
import pickle
import numpy as np
from data_process_script.data_process import Processor

from data_loader_script.data_loader_bert_base import EleDataset


from model_script.Bert_base_model import MyBertBase_Model
from train_script_for_bert_base import train

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers.optimization import get_cosine_schedule_with_warmup, AdamW 
import torch.optim as optim
from util_script.optimizer_wf import RAdam
import torch
import random
import sys
import warnings
from torch.utils.tensorboard import SummaryWriter
from vail_test_data_script.bert_base import eval_link
import shutil

from util_script.NER_F1 import *

warnings.filterwarnings('ignore')

def seed_everything(seed=1029):
    '''
    设置整个开发环境的seed
    :param seed:
    :return:
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True # 这里设置为true是为了保证算法的可复现性

def get_three_data(train_doc_num,test_doc_num):
    if(train_doc_num==55 and test_doc_num==14):
        fn1 = '/data2/fwang/spatial_relation_extract/code/data/data_process_pkl/ori_data_summary_55_14/train_data_summary.pkl'
        fn2 = '/data2/fwang/spatial_relation_extract/code/data/data_process_pkl/ori_data_summary_55_14/train_data_id_summary.pkl'
        fn3 = '/data2/fwang/spatial_relation_extract/code/data/data_process_pkl/ori_data_summary_55_14/vail_data_summary.pkl'
        fn4 = '/data2/fwang/spatial_relation_extract/code/data/data_process_pkl/ori_data_summary_55_14/vail_data_id_summary.pkl'
        fn5 = '/data2/fwang/spatial_relation_extract/code/data/data_process_pkl/ori_data_summary_55_14/test_data_summary.pkl'
        fn6 = '/data2/fwang/spatial_relation_extract/code/data/data_process_pkl/ori_data_summary_55_14/test_data_id_summary.pkl'
    elif(train_doc_num==59 and test_doc_num==16):
        fn1 = '/data2/fwang/spatial_relation_extract/code/data/data_process_pkl/ori_data_summary_59_16/train_data_summary.pkl'
        fn2 = '/data2/fwang/spatial_relation_extract/code/data/data_process_pkl/ori_data_summary_59_16/train_data_id_summary.pkl'
        fn3 = '/data2/fwang/spatial_relation_extract/code/data/data_process_pkl/ori_data_summary_59_16/vail_data_summary.pkl'
        fn4 = '/data2/fwang/spatial_relation_extract/code/data/data_process_pkl/ori_data_summary_59_16/vail_data_id_summary.pkl'
        fn5 = '/data2/fwang/spatial_relation_extract/code/data/data_process_pkl/ori_data_summary_59_16/test_data_summary.pkl'
        fn6 = '/data2/fwang/spatial_relation_extract/code/data/data_process_pkl/ori_data_summary_59_16/test_data_id_summary.pkl'
    else:
        raise Exception("没有该指定的数据集！\n 请检查是否是如下之一：\n1、训练集55测试集14\n2、训练集59测试集16")
    with open(fn1, 'rb') as f:  
        train_oridocdic = pickle.load(f)
    with open(fn2, 'rb') as f:  
        train_docid2docname = pickle.load(f)

    with open(fn3, 'rb') as f:  
        vail_oridocdic = pickle.load(f)
    with open(fn4, 'rb') as f:  
        vail_docid2docname = pickle.load(f)

    with open(fn5, 'rb') as f:  
        test_oridocdic = pickle.load(f)
    with open(fn6, 'rb') as f:  
        test_docid2docname = pickle.load(f)
    return train_oridocdic,train_docid2docname,vail_oridocdic,vail_docid2docname,test_oridocdic,test_docid2docname

def run():
    seed_everything()
    """train the model"""
    # set the logger
    # utils.set_logger(config.log_dir)
    # config.learning_rate = eval(sys.argv[1])
    # config.batch_size = int(sys.argv[2])
    # config.save_train_fst_model_dir += str(sys.argv[1])
    # config.save_train_fst_model_dir += str(sys.argv[2])
    logging.info("device: {}".format(config.device))
    # 处理数据，分离文本和标签
    # processor = Processor(config)
    
    # processor.process()
    train_oridocdic,train_docid2docname,vail_oridocdic,vail_docid2docname,test_oridocdic,test_docid2docname=get_three_data(55,14)
    logging.info("--------Process Done!--------")
    # 分离出验证集
    # word_train, word_dev, label_train, label_dev = load_dev('train')
    # build dataset
    # 这里划分一下训练集 将其中的一部分作为测试集
    # vail_oridocdic=dict()
    # train_oridocdic=dict()
    # for k,v in processor.train_oridocdic.items():
    #     if(k<=44):
    #         train_oridocdic[k]=v
    #     else:
    #         vail_oridocdic[k]=v

    train_dataset = EleDataset(train_oridocdic,train_docid2docname,config)
    dev_dataset = EleDataset(vail_oridocdic,vail_docid2docname, config)
    test_dataset = EleDataset(test_oridocdic,test_docid2docname, config)
    logging.info("--------Dataset Build!--------")
    # get dataset size
    train_size = len(train_dataset)
    # print(train_size)
    # build data_loader
    # print(len(train_dataset))
    # print(config.batch_size)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True, collate_fn=train_dataset.collate_fn)
    # print(len(train_loader))
    dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size,
                            shuffle=False, collate_fn=dev_dataset.collate_fn)

    test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                            shuffle=False, collate_fn=test_dataset.collate_fn)
    logging.info("--------Get Dataloader!--------")
    # Prepare model
    device = config.device
    model=MyBertBase_Model()
    model.to(device)
    # Prepare optimizer
    # if config.full_fine_tuning:
    #     # model.named_parameters(): [bert, classifier]
    #     bert_optimizer = list(model.bertmodel.named_parameters())
    #     classifier_optimizer = list(model.outlin.named_parameters())
    #     no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    #     #可以精确控制bert中的每个参数的权重衰减
    #     optimizer_grouped_parameters = [
    #         {'params': [p for n, p in bert_optimizer if not any(nd in n for nd in no_decay)],
    #          'weight_decay': config.weight_decay},
    #         {'params': [p for n, p in bert_optimizer if any(nd in n for nd in no_decay)],
    #          'weight_decay': 0.0},
    #         {'params': [p for n, p in classifier_optimizer if not any(nd in n for nd in no_decay)],
    #          'lr': config.learning_rate * 5, 'weight_decay': config.weight_decay},
    #         {'params': [p for n, p in classifier_optimizer if any(nd in n for nd in no_decay)],
    #          'lr': config.learning_rate * 5, 'weight_decay': 0.0}
    #     ]
    # # only fine-tune the head classifier
    # else:
    #     param_optimizer = list(model.outlin.named_parameters())
    #     optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer]}]
    # optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate, correct_bias=False)
    # optimizer=optim.Adam([{'params': model.parameters()}],
    #             lr=config.learning_rate,
    #             betas=(0.9, 0.999),batch_sizerning_rate)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    train_steps_per_epoch = train_size // config.batch_size
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=train_steps_per_epoch,
                                                num_training_steps=config.epoch_num * train_steps_per_epoch)
    #  * (config.epoch_num // 10)
    # Train the model
    logging.info("--------Start Training!--------")
    if(not os.path.exists(config.train_log_dir)):
        os.makedirs(config.train_log_dir)
    else :
        shutil.rmtree(config.train_log_dir)
    if(not os.path.exists(config.test_log_dir)):
        os.makedirs(config.test_log_dir)
    else:
        shutil.rmtree(config.test_log_dir)
    train_writer = SummaryWriter(log_dir=config.train_log_dir)
    test_writer = SummaryWriter(log_dir=config.test_log_dir)
    # train( model, optimizer, train_loader, dev_loader,scheduler, config.model_dir)
    train( model, optimizer, train_loader,eval_fun=f1_score,dev_loader=dev_loader,scheduler=scheduler,train_writer=train_writer,test_writer=test_writer)
    eval_link(test_loader)

if __name__ == '__main__':
    run()