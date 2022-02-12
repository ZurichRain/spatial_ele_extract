'''
学习一个生成模型 
将关系预测看作是一个生成任务 
输入是一句话[tok1,tok2...] 
输出是[link_stype,tr,tj,ld] 
linkstype=[no_link,qslink,olink,movelink] 
tr=[None,] 
tj=[None,]
ld=[None,]
'''
import sys
import os
sys.path.append('./code/')
from transformers import BertTokenizer,BertModel,BertTokenizerFast,BertPreTrainedModel,get_linear_schedule_with_warmup
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch.nn as nn
import torch
import config_script.config as config
from allennlp.modules.span_extractors import SelfAttentiveSpanExtractor

# T5词表大小是32128

tokenizer = T5Tokenizer.from_pretrained(config.T5base_pathname)
model = T5ForConditionalGeneration.from_pretrained(config.T5base_pathname)
input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
# print(input_ids,labels)
print(labels.size())
outputs = model(input_ids=input_ids, labels=labels)
loss_fun=nn.CrossEntropyLoss()
print(loss_fun(outputs.logits.view(-1,32128),labels.view(-1)))
print(outputs.logits.size())
print(outputs.loss)
# token_span=tokenizer.encode_plus("The <extra_id_0> walks in <extra_id_1> park", add_special_tokens=False,return_offsets_mapping=True)
# print(tokenizer.decode(outputs[0], skip_special_tokens=False))


# input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
# labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
# outputs = model(input_ids=input_ids, labels=labels)
# loss = outputs.loss
# logits = outputs.logits
# print(loss)
# print(logits)

class MyT5_Model(nn.Module):
    # config_class = RobertaConfig

    def __init__(self):
        super(MyT5_Model,self).__init__()

        # self.y_num = 6
        # self.bertmodel = bert_model
        
        self.num_labels1=4
        # self.hidden_size=256
        # self.bertmodel=bert_model
        self.bertmodel=BertModel.from_pretrained(config.bert_pathname)
        # self.bertmodel=BertModel(config)
        self.tanh=nn.Tanh()
        self.outlin1=nn.Linear(3*768,self.num_labels1)
        torch.nn.init.xavier_uniform_(self.outlin1.weight)
        self.drop1=nn.Dropout(p=0.15) #加上dropout 没有任何的作用
        self.loss1 = nn.CrossEntropyLoss(
            # weight=torch.FloatTensor([0.1,1.0])
            # ignore_index=0
        )
        self.span_extractor=SelfAttentiveSpanExtractor(input_dim=768)
        # self.span_extractor2=SelfAttentiveSpanExtractor(input_dim=768)



    def forward(self, ctrain_x_tok_ids,batch_e1_mask,batch_e2_mask,batch_tr_mask,ctrain_y):
        #print(data_x.shape)
        # bertout = self.bertmodel(input_ids=ctrain_x_tok_ids,attention_mask=ctrain_x_tok_mask,token_type_ids=ctrain_x_tok_segment)
        bertout=self.bertmodel(ctrain_x_tok_ids)
        # bert_enc = outputs[0]
        wemb=bertout[0]
        cure1_emb=self.span_extractor(wemb,batch_e1_mask.unsqueeze(1))
        cure2_emb=self.span_extractor(wemb,batch_e2_mask.unsqueeze(1))
        curetr_emb=self.span_extractor(wemb,batch_tr_mask.unsqueeze(1))
        activate_emb=torch.cat((cure1_emb,cure2_emb,curetr_emb),dim=-1)
        # activate_emb=torch.cat((cure1_emb,cure2_emb,curtr_emb),dim=-1)
        # activate_emb=self.tanh(activate_emb)
        activate_emb=self.drop1(activate_emb) 
        activate_emb=self.outlin1(activate_emb)
        # print(activate_emb.size())
        # exit()
        # wemb=self.sft(wemb)
        # prey=wemb.squeeze(0)
        
        wemb1=activate_emb.view(-1,self.num_labels1)
        ctrain_y=ctrain_y.view(-1)
        l1=self.loss1(wemb1,ctrain_y)
        return l1,wemb1