import sys
import os
sys.path.append('./code/')
from torch.nn.modules import loss
from transformers import BertTokenizer,BertModel
import torch.nn as nn
import torch
from torchcrf import CRF
import config_script.config as config
from allennlp.modules.span_extractors import SelfAttentiveSpanExtractor
#这个模型是使用了span——attention的
# 又是失败的一个模型 不收敛的模型 train上只能到 57 验证集上只能到47

class FocalLoss(nn.Module):

    def __init__(self, weight=None, reduction='mean', gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss(weight=weight, reduction=reduction)

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()
class MyQSlink_Model(nn.Module):
    # config_class = RobertaConfig

    def __init__(self):
        super(MyQSlink_Model,self).__init__()
        
        self.ele_num_labels=17
        self.qs_role_num_labels=7
        self.bertmodel=BertModel.from_pretrained(config.bert_pathname)
        self.outlin_ele=nn.Linear(768,self.ele_num_labels)
        self.outlin_qs_role=nn.Linear(768,self.qs_role_num_labels)
        # torch.nn.init.xavier_uniform_(self.outlin1.weight)
        self.ele_crf = CRF(self.ele_num_labels,batch_first=True)
        self.role_crf = CRF(self.qs_role_num_labels,batch_first=True)
        self.drop = nn.Dropout(p=0.1) #加上dropout 没有任何的作用
        self.loss = nn.CrossEntropyLoss()
        self.sft = nn.Softmax()



    def forward(self, datas,ele_labels,role_labels,seq_mask):
        #print(data_x.shape)
        # bertout = self.bertmodel(input_ids=ctrain_x_tok_ids,attention_mask=ctrain_x_tok_mask,token_type_ids=ctrain_x_tok_segment)
        bertout=self.bertmodel(datas)
        # bert_enc = outputs[0]
        wemb=bertout[0]
        # ele_emb=self.sft(self.outlin_ele(wemb))
        ele_emb=self.outlin_ele(wemb)
        # qs_role_emb=self.sft(self.outlin_qs_role(wemb))
        qs_role_emb=self.outlin_qs_role(wemb)

        ele_loss=self.ele_crf(ele_emb,ele_labels,mask=seq_mask)
        ele_decode=self.ele_crf.decode(ele_emb)
        role_loss=self.role_crf(qs_role_emb,role_labels,mask=seq_mask)
        role_decode = self.role_crf.decode(qs_role_emb)
        return -ele_loss-role_loss,ele_decode,role_decode

class MyOlink_Model(nn.Module):
    # config_class = RobertaConfig

    def __init__(self):
        super(MyQSlink_Model,self).__init__()
        
        self.ele_num_labels=17
        self.qs_role_num_labels=7
        self.bertmodel=BertModel.from_pretrained(config.bert_pathname)
        self.outlin_ele=nn.Linear(768,self.ele_num_labels)
        self.outlin_qs_role=nn.Linear(768,self.qs_role_num_labels)
        # torch.nn.init.xavier_uniform_(self.outlin1.weight)
        self.ele_crf = CRF(self.ele_num_labels,batch_first=True)
        self.role_crf = CRF(self.qs_role_num_labels,batch_first=True)
        self.drop = nn.Dropout(p=0.1) #加上dropout 没有任何的作用
        self.loss = nn.CrossEntropyLoss()
        self.sft = nn.Softmax()



    def forward(self, datas,ele_labels,role_labels,seq_mask):
        #print(data_x.shape)
        # bertout = self.bertmodel(input_ids=ctrain_x_tok_ids,attention_mask=ctrain_x_tok_mask,token_type_ids=ctrain_x_tok_segment)
        bertout=self.bertmodel(datas)
        # bert_enc = outputs[0]
        wemb=bertout[0]
        # ele_emb=self.sft(self.outlin_ele(wemb))
        ele_emb=self.outlin_ele(wemb)
        # qs_role_emb=self.sft(self.outlin_qs_role(wemb))
        qs_role_emb=self.outlin_qs_role(wemb)

        ele_loss=self.ele_crf(ele_emb,ele_labels,mask=seq_mask)
        ele_decode=self.ele_crf.decode(ele_emb)
        role_loss=self.role_crf(qs_role_emb,role_labels,mask=seq_mask)
        role_decode = self.role_crf.decode(qs_role_emb)
        return -ele_loss-role_loss,ele_decode,role_decode


class MyMovelink_Model(nn.Module):
    # config_class = RobertaConfig

    def __init__(self):
        super(MyQSlink_Model,self).__init__()
        
        self.ele_num_labels=17
        self.qs_role_num_labels=7
        self.bertmodel=BertModel.from_pretrained(config.bert_pathname)
        self.outlin_ele=nn.Linear(768,self.ele_num_labels)
        self.outlin_qs_role=nn.Linear(768,self.qs_role_num_labels)
        # torch.nn.init.xavier_uniform_(self.outlin1.weight)
        self.ele_crf = CRF(self.ele_num_labels,batch_first=True)
        self.role_crf = CRF(self.qs_role_num_labels,batch_first=True)
        self.drop = nn.Dropout(p=0.1) #加上dropout 没有任何的作用
        self.loss = nn.CrossEntropyLoss()
        self.sft = nn.Softmax()



    def forward(self, datas,ele_labels,role_labels,seq_mask):
        #print(data_x.shape)
        # bertout = self.bertmodel(input_ids=ctrain_x_tok_ids,attention_mask=ctrain_x_tok_mask,token_type_ids=ctrain_x_tok_segment)
        bertout=self.bertmodel(datas)
        # bert_enc = outputs[0]
        wemb=bertout[0]
        # ele_emb=self.sft(self.outlin_ele(wemb))
        ele_emb=self.outlin_ele(wemb)
        # qs_role_emb=self.sft(self.outlin_qs_role(wemb))
        qs_role_emb=self.outlin_qs_role(wemb)

        ele_loss=self.ele_crf(ele_emb,ele_labels,mask=seq_mask)
        ele_decode=self.ele_crf.decode(ele_emb)
        role_loss=self.role_crf(qs_role_emb,role_labels,mask=seq_mask)
        role_decode = self.role_crf.decode(qs_role_emb)
        return -ele_loss-role_loss,ele_decode,role_decode

