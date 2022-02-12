import sys
import os
sys.path.append('./code/')
import torch
import config_script.config as config
# from util_script.metrics import f1_score_3
from util_script.NER_F1 import *
def eval_link(test_loader):
    model = torch.load(config.save_train_model_file)
    model.eval()
    test_losses = 0
    # pre_tok_lab_lis = []
    # pre_tok_lis=[]
    # pre_ori_st_ed=[] #保存每个预测标签在原文中的st和ed
    # pre_ori_link=[]
    ele_pre_labels_lis=[]
    ele_true_labels_lis=[]
    with torch.no_grad():
        for idx, batch_samples in enumerate(test_loader):
            cbatch=batch_samples
            # print(batch_labels)
            # batch_masks = batch_data.gt(0)  # get padding mask
            # compute model output and loss
            # loss,prey1 = model(batch_data,batch_e1_mask,batch_e2_mask,batch_tr_mask,batch_labels)
            loss,prey1=model(**cbatch)
            # exit()
            batch_lab_masks=cbatch['seq_mask'].view(-1) == 1
            prey1_lab=torch.argmax(prey1,dim=-1)
            ele_pre_labels_lis += prey1_lab.to('cpu').numpy().tolist()
            ele_true_labels_lis += cbatch['ele_labels'].view(-1)[batch_lab_masks].to('cpu').numpy().tolist()
            test_losses += loss.item()

    # assert len(prey_lis) == len(truy_lis)
    # for orilin in pre_ori_link:
    #     print(orilin)
    metrics = {}
    metrics['all_ele_f1'],metrics['ele_f1']=f1_score(ele_true_labels_lis,ele_pre_labels_lis,config.ele_labels,config.spatial_ele_label2id,mode='muti')
    metrics['loss'] = float(test_losses) / len(test_loader)

    print('testf1: ',metrics['ele_f1'])
    print('loss: ',metrics['loss'])
    if(os.path.exists(config.save_train_result_dir)):
        with open (config.save_train_result_file,'w')as f:
            f.write(str(metrics)+'\n')
    else:
        os.makedirs(config.save_train_result_dir)
        with open (config.save_train_result_file,'w')as f:
            f.write(str(metrics)+'\n')
    
    