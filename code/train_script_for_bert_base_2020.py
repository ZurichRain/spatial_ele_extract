import sys
import os
sys.path.append('./code/')
import torch
import logging
import torch.nn as nn
from tqdm import tqdm

import config_script.config as config
import numpy as np
from model_script.Bert_base_model_2020 import MyQSlink_Model,MyOlink_Model,MyMovelink_Model
from transformers import BertTokenizer,BertTokenizerFast
from util_script.NER_F1 import *
# from my_f1_score import f1_score

# def semple_eval(model,data_loader):
#     model.eval()
#     with torch.no_grad():
#         for idx, batch_samples in enumerate(data_loader):
#             batch_data, batch_token_starts, batch_tags = batch_samples

# 一个项目
# log
# model_script
# model
# data_process_script
# data_loader_script
# test_code_script
# config_script
# util_script
# vail_test_data_script
# run.py
# train.py
# run_train.sh
# run_vail_test_data.sh


def train_epoch(train_loader, model, optimizer, epoch , eval_fun ,scheduler=None,train_writer=None):
    # set model to training mode
    model.train()
    # step number in one epoch: 336
    train_losses = 0
    ele_pre_labels_lis=[]
    ele_true_labels_lis=[]

    role_pre_labels_lis=[]
    role_true_labels_lis=[]
    
    for idx, batch_samples in enumerate(tqdm(train_loader)):
        # batch_data, batch_e1_mask,batch_e2_mask,batch_tr_mask,batch_labels = batch_samples
        cbatch=batch_samples
        # print(batch_labels)
        # batch_masks = batch_data.gt(0)  # get padding mask
        # compute model output and loss
        # loss,prey1 = model(batch_data,batch_e1_mask,batch_e2_mask,batch_tr_mask,batch_labels)
        loss,ele_decode,role_decode=model(**cbatch)
        # exit()
        ele_pre_labels_lis += ele_decode
        ele_true_labels_lis += cbatch['ele_labels'].to('cpu').numpy().tolist()

        role_pre_labels_lis += role_decode
        role_true_labels_lis += cbatch['role_labels'].to('cpu').numpy().tolist()
        

        train_losses += loss.item()
        # clear previous gradients, compute gradients of all variables wrt loss
        optimizer.zero_grad()
        # print([(n,p) for n,p in list(model.named_parameters()) if 'outlin_tj' in n])
        # for n,p in list(model.named_parameters()):
        #     print(n,p,p.grad)
        #     print('*'*100)
        #     break
        #         print(n, p , p.grad ,p.requires_grad)
        # gradient clipping 可以解决梯度爆炸或者梯度消失的问题
        # nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=config.clip_grad)
        # performs updates using calculated gradients
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
    # print(prey_lis)
    # print(truy_lis)
    # one_dim_ele_pre=[]
    # for i in ele_pre_labels_lis:
    #     for j in i:
    #         one_dim_ele_pre.append(j)

    # one_dim_ele_true=[]
    # for i in ele_true_labels_lis:
    #     for j in i:
    #         one_dim_ele_true.append(j)
    train_ele_epoch_f1=eval_fun(ele_true_labels_lis,ele_pre_labels_lis,config.ele_labels,config.spatial_ele_label2id)
    train_role_epoch_f1 = eval_fun(role_true_labels_lis,role_pre_labels_lis,config.qsrole_labels,config.spatial_qsrole_label2id)


    logging.info("Epoch: {}, train_ele_epoch_f1: {}".format(epoch,train_ele_epoch_f1))
    logging.info("Epoch: {}, train_role_epoch_f1: {}".format(epoch,train_role_epoch_f1))
    train_loss = float(train_losses) / len(train_loader)
    logging.info("Epoch: {}, train loss: {}".format(epoch, train_loss))
    if train_writer is not None:
        train_writer.add_scalar('loss', train_loss, epoch)
        train_writer.add_scalar('F1', train_ele_epoch_f1, epoch)

def train( model, optimizer,train_loader, eval_fun,dev_loader=None, scheduler=None, model_dir=None,train_writer=None,test_writer=None):
    """train the model and test model performance"""
    # reload weights from restore_dir if specified
    if model_dir is not None and config.load_before:
        if(config.sub_model_name=='qslink_model'):
            model = MyQSlink_Model.from_pretrained(model_dir)
        elif(config.sub_model_name=='olink_model'):
            model = MyOlink_Model.from_pretrained(model_dir)
        elif(config.sub_model_name=='movelink_model'):
            model = MyMovelink_Model.from_pretrained(model_dir)
        else:
            raise Exception("子模型只能是以下三个模型之一：\n1、qslink_model\n2、olink_model\n3、movelink_model")
       
        model.to(config.device)
        logging.info("--------Load model from {}--------".format(model_dir))
    best_val_f1 = 0.0
    patience_counter = 0
    # start training
    # optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    for epoch in range(1, config.epoch_num + 1):
        train_epoch(train_loader, model, optimizer, epoch ,eval_fun , scheduler,train_writer)
        if dev_loader is None:
            pass
        else:
            # pass
            val_metrics = evaluate(dev_loader, model,epoch,eval_fun,test_writer=test_writer)
            val_ele_f1 = val_metrics['ele_f1']
            val_role_f1 = val_metrics['role_f1']
            logging.info("Epoch: {}, dev loss: {}, ele_f1 score: {}".format(epoch, val_metrics['loss'], val_ele_f1))
            logging.info("Epoch: {}, dev loss: {}, role_f1 score: {}".format(epoch, val_metrics['loss'], val_role_f1))
            improve_f1 = val_ele_f1 - best_val_f1
            
            if improve_f1 > 1e-5:
                best_val_f1 = val_ele_f1
                if(os.path.exists(config.save_train_model_dir)):
                    torch.save(model,config.save_train_model_file)
                else:
                    os.makedirs(config.save_train_model_dir)
                    torch.save(model,config.save_train_model_file)
                logging.info("--------Save best model!--------")
                if improve_f1 < config.patience: #如果增加的效果小于给定的阈值 那么就认为是没有增加
                    patience_counter += 1
                else:
                    patience_counter = 0
            else:
                patience_counter += 1
            # Early stopping and logging best f1
            if (patience_counter >= config.patience_num and epoch > config.min_epoch_num) or epoch == config.epoch_num:
                logging.info("Best val f1: {}".format(best_val_f1))
                break
            # if(epoch==config.epoch_num):
            #     break
    logging.info("Training Finished!")

def evaluate(dev_loader, model, epoch,eval_fun,mode='dev',test_writer=None):
    # set model to evaluation mode
    model.eval()

    # tokenizer = BertTokenizer.from_pretrained(config.bert_model, do_lower_case=True, skip_special_tokens=True)
    # tokenizer=BertTokenizerFast.from_pretrained(config.bert_name)
    # id2label = config.id2label
    # true_tags = []
    # pred_tags = []
    # sent_data = []

    dev_losses = 0
    ele_pre_labels_lis=[]
    ele_true_labels_lis=[]

    role_pre_labels_lis=[]
    role_true_labels_lis=[]
    
    for idx, batch_samples in enumerate(tqdm(dev_loader)):
        # batch_data, batch_e1_mask,batch_e2_mask,batch_tr_mask,batch_labels = batch_samples
        cbatch=batch_samples
        # print(batch_labels)
        # batch_masks = batch_data.gt(0)  # get padding mask
        # compute model output and loss
        # loss,prey1 = model(batch_data,batch_e1_mask,batch_e2_mask,batch_tr_mask,batch_labels)
        loss,ele_decode,role_decode=model(**cbatch)

        ele_pre_labels_lis += ele_decode
        ele_true_labels_lis += cbatch['ele_labels'].to('cpu').numpy().tolist()

        role_pre_labels_lis += role_decode
        role_true_labels_lis += cbatch['role_labels'].to('cpu').numpy().tolist()
        

        dev_losses += loss.item()
        # clear previous gradients, compute gradients of all variables wrt loss
    metrics = {}
    metrics['ele_f1']=eval_fun(ele_true_labels_lis,ele_pre_labels_lis,config.ele_labels,config.spatial_ele_label2id)
    metrics['role_f1']=eval_fun(role_true_labels_lis,role_pre_labels_lis,config.qsrole_labels,config.spatial_qsrole_label2id)
    # if mode == 'dev':
    #     f1 = f1_score_1(true_tags, pred_tags, mode)
    #     metrics['f1'] = f1
    # else:
    #     # bad_case(true_tags, pred_tags, sent_data)
    #     f1_labels, f1 = f1_score_1(true_tags, pred_tags, mode)
    #     metrics['f1_labels'] = f1_labels
    #     metrics['f1'] = f1
    metrics['loss'] = float(dev_losses) / len(dev_loader)
    if(test_writer is not None):
        test_writer.add_scalar('F1', metrics['ele_f1'], epoch)
        test_writer.add_scalar('loss', metrics['loss'], epoch)
    return metrics
