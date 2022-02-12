import sys
import os
sys.path.append('./code/')
from torch.nn.modules import loss
from transformers import BertTokenizer,BertModel,BertTokenizerFast,BertPreTrainedModel,get_linear_schedule_with_warmup
import torch.nn as nn
import torch
from torchcrf import CRF
import config_script.config as config
from allennlp.modules.span_extractors import SelfAttentiveSpanExtractor


class MyBertBase_Model(nn.Module):
    # config_class = RobertaConfig

    def __init__(self):
        super(MyBertBase_Model,self).__init__()

        # self.y_num = 6
        # self.bertmodel = bert_model
        
        self.num_labels1=17
        # self.hidden_size=256
        # self.bertmodel=bert_model
        self.bertmodel=BertModel.from_pretrained(config.bert_pathname)
        # self.bertmodel=BertModel(config)
        self.ele_prb=nn.Linear(768,self.num_labels1)
        torch.nn.init.xavier_uniform_(self.ele_prb.weight)
        self.drop1=nn.Dropout(p=0.15) #加上dropout 没有任何的作用
        self.loss1 = nn.CrossEntropyLoss(
            # weight=torch.FloatTensor([0.1,1.0])
            # ignore_index=0
        )



    def forward(self,  datas,ele_labels,seq_mask):
        #print(data_x.shape)
        bertout=self.bertmodel(datas)
        # bert_enc = outputs[0]
        wemb=bertout[0]
        wemb=self.drop1(wemb) 
        ele_pre_prb=self.ele_prb(wemb)

        seq_mask=seq_mask.view(-1) == 1
        activate_emb=ele_pre_prb.view(-1,self.num_labels1)[seq_mask]
        
        wemb1=activate_emb.view(-1,self.num_labels1)
        ctrain_y=ele_labels.view(-1)[seq_mask]
        l1=self.loss1(wemb1,ctrain_y)
        return l1,wemb1