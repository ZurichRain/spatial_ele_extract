import sys
import os
sys.path.append('./code/')
import torch
import config_script.config as config
from util_script.NER_F1 import *
from sklearn.metrics import classification_report

def eval_link(test_loader):
    model = torch.load(config.save_train_model_file)
    model.eval()
    test_losses = 0
    # pre_tok_lab_lis = []
    # pre_tok_lis=[]
    # pre_ori_st_ed=[] #保存每个预测标签在原文中的st和ed
    # pre_ori_link=[]
    with torch.no_grad():
        ele_pre_labels_lis=[]
        ele_true_labels_lis=[]

        role_pre_labels_lis=[]
        role_true_labels_lis=[]
        
        for idx, batch_samples in enumerate(test_loader):
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
            

            test_losses += loss.item()

    # assert len(prey_lis) == len(truy_lis)
    # for orilin in pre_ori_link:
    #     print(orilin)
    metrics = {}
    metrics['all_ele_f1'],metrics['ele_f1']=f1_score(ele_true_labels_lis,ele_pre_labels_lis,config.ele_labels,config.spatial_ele_label2id,mode='muti')
    metrics['all_role_f1'],metrics['role_f1']=f1_score(role_true_labels_lis,role_pre_labels_lis,config.qsrole_labels,config.spatial_qsrole_label2id,mode='muti')
    metrics['loss'] = float(test_losses) / len(test_loader)

    print('test_ele_f1: ',metrics['ele_f1'])
    print('loss: ',metrics['loss'])
    if(os.path.exists(config.save_train_result_dir)):
        with open (config.save_train_result_file,'w')as f:
            f.write(str(metrics)+'\n')
    else:
        os.makedirs(config.save_train_result_dir)
        with open (config.save_train_result_file,'w')as f:
            f.write(str(metrics)+'\n')
    
    