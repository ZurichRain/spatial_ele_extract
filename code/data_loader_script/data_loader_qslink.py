'''
构建空间元素的标签格式 joint qslink role
即 需要构建 ele_label=[] 和 role_label=[]

'''
import sys
import os
sys.path.append('./code/')
from data_process_script.data_process import *
from util_script.mata_data_calss import *
from transformers import BertTokenizerFast
import torch
from torch.utils.data import Dataset

from sklearn.metrics import recall_score,precision_score,f1_score,confusion_matrix,roc_curve,accuracy_score

class EleDataset(Dataset):
    def __init__(self, oridocdic,docid2docname, config, tok_pad_idx=0, label_pad_idx=-1):
        # bert_name = 'bert-base-uncased'
        # config.train_CP_dir config.train_ANC_dir config.train_RFC_dir
        self.config = config
        self.tokenizer = BertTokenizerFast.from_pretrained(config.bert_pathname)
        self.oridocdic = oridocdic
        self.dataset = self.preprocess(self.oridocdic,docid2docname)
        self.device = config.device

    
    def preprocess(self,oridocdic,docid2docname):
        # oridocdic,docid2docname=get_ori_doc_info(self.config.train_CP_dir)
        all_seq_max_len=self.get_seq_tok_all_doc(oridocdic,self.tokenizer,docid2docname)# 第一步首先获取所有句子的token list
        # self.get_seq_tok_lab_all_doc(oridocdic,self.tokenizer)
        self.get_all_doc_ele_labels(oridocdic,self.tokenizer)
        self.get_all_doc_qs_role_labels(oridocdic,self.tokenizer)
        # all_seq_lis=[]
        all_link_seq_lis=[]
        # all_link_lis=[]
        all_ele_labels_lis=[]
        all_qs_role_labels_lis=[]
        for k,v in self.oridocdic.items():
            all_link_seq_lis+=v.seqtokidslis
            all_ele_labels_lis+=v.all_ele_labels
            all_qs_role_labels_lis+=v.all_qs_role_labels
        data=[]
        for seq,ele_label,qs_role_label in zip(all_link_seq_lis,all_ele_labels_lis,all_qs_role_labels_lis):
            data.append((seq,ele_label,qs_role_label))
        return data
    
    def get_seq_tok_one_doc(self,doc,tokenizer):
        curseqtoklis=[]
        curseqtokidslis=[]
        doc_max_seq_len=0
        for seq in doc.seqlis:
            curseqtok=tokenizer.tokenize(seq)# token 之后就会把空格去掉不知道会不会有影响
            doc_max_seq_len=max(doc_max_seq_len,len(curseqtok))
            curseqtoklis.append(curseqtok)
            curseqtokidslis.append(tokenizer.convert_tokens_to_ids(curseqtok))
        
        doc.seqtoklis=curseqtoklis
        doc.seqtokidslis=curseqtokidslis
        return doc_max_seq_len

    def get_seq_tok_all_doc(self,oridocdic,tokenizer,docid2docname):
        #对于每一篇文章得到它的seqtoken
        all_max_seq_len=0
        for k,v in oridocdic.items():
            curmaxlen=self.get_seq_tok_one_doc(v,tokenizer)
            # data = {}
            # for idx,seq in enumerate(v.seqtoklis):
            #     data[str(idx)]=[seq,v.seqtokidslis[idx]]
            # data_df = pd.DataFrame(data,index=[0,1])
            # data_df.to_csv('/data2/fwang/baseline/data/my_process_data/test_toks/'+docid2docname[k].split('/')[-1][:-4]+'_seqtoks.csv')
            all_max_seq_len=max(all_max_seq_len,curmaxlen)
            # print(v.seqtokidslis)
            # print(curmaxlen)
        return all_max_seq_len


    def legal(self,ele):
        if(int(ele.start)==-1 or not hasattr(ele,'seqstid')):
            return False
        return True
    def get_ele_st_ed(self,ele,char2tok_span):
        charst=ele.seqstid[1]
        chared=ele.seqedid[1]
        ele_tok_st_ed_lis=[]
        for i in range(charst,chared):
            if(char2tok_span[i][0] not in ele_tok_st_ed_lis and char2tok_span[i][0]!=-1 and char2tok_span[i][1]!=-1):
                ele_tok_st_ed_lis.append(char2tok_span[i][0])
        return ele_tok_st_ed_lis
    def get_one_doc_ele_labels(self,doc,tokenizer):
        one_doc_ele_labels=[[0 for _ in seq] for seq  in doc.seqtokidslis]
        spatial_ele_label2id=config.spatial_ele_label2id
        seqs_char2tok_span=self.get_char2tok_spanlis_one_doc(doc,tokenizer)
        for celelisname in ele_lis_name:
            bidx=spatial_ele_label2id['B-'+celelisname[:-4]]
            iidx=spatial_ele_label2id['I-'+celelisname[:-4]]
            cur_ele_lis=getattr(doc,celelisname)
            for ele in cur_ele_lis:
                if(self.legal(ele)):
                    curseqid=ele.seqstid[0]
                    ele_tok_st_ed_lis=self.get_ele_st_ed(ele,seqs_char2tok_span[curseqid])
                    one_doc_ele_labels[curseqid][ele_tok_st_ed_lis[0]]=bidx
                    for ciidx in ele_tok_st_ed_lis[1:]:
                        one_doc_ele_labels[curseqid][ciidx]=iidx
        
        doc.all_ele_labels=one_doc_ele_labels
        

    def get_all_doc_ele_labels(self,oridocdic,tokenizer):
        for k,v in oridocdic.items():
            self.get_one_doc_ele_labels(v,tokenizer)
    
    def get_one_doc_qs_role_labels(self,doc,tokenizer):
        one_doc_qs_role_labels=[[0 for _ in seq] for seq  in doc.seqtokidslis]
        seqs_char2tok_span=self.get_char2tok_spanlis_one_doc(doc,tokenizer)
        for lin in doc.qslink_lis:
            if(len(lin.trajector)>0):
                tj_ele=doc.id2obj[lin.trajector]
                if(self.legal(tj_ele)):
                    curseqid=tj_ele.seqstid[0]
                    ele_tok_st_ed_lis=self.get_ele_st_ed(tj_ele,seqs_char2tok_span[curseqid])
                    one_doc_qs_role_labels[curseqid][ele_tok_st_ed_lis[0]]=1
                    for ciidx in ele_tok_st_ed_lis[1:]:
                        one_doc_qs_role_labels[curseqid][ciidx]=2
            if(len(lin.landmark)>0):
                ld_ele=doc.id2obj[lin.trajector]
                if(self.legal(ld_ele)):
                    curseqid=ld_ele.seqstid[0]
                    ele_tok_st_ed_lis=self.get_ele_st_ed(ld_ele,seqs_char2tok_span[curseqid])
                    one_doc_qs_role_labels[curseqid][ele_tok_st_ed_lis[0]]=3
                    for ciidx in ele_tok_st_ed_lis[1:]:
                        one_doc_qs_role_labels[curseqid][ciidx]=4
            if(len(lin.trigger)>0):
                tr_ele=doc.id2obj[lin.trajector]
                if(self.legal(tr_ele)):
                    curseqid=tr_ele.seqstid[0]
                    ele_tok_st_ed_lis=self.get_ele_st_ed(tr_ele,seqs_char2tok_span[curseqid])
                    one_doc_qs_role_labels[curseqid][ele_tok_st_ed_lis[0]]=5
                    for ciidx in ele_tok_st_ed_lis[1:]:
                        one_doc_qs_role_labels[curseqid][ciidx]=6

        doc.all_qs_role_labels=one_doc_qs_role_labels

    def get_all_doc_qs_role_labels(self,oridocdic,tokenizer):
        for k,v in oridocdic.items():
            self.get_one_doc_qs_role_labels(v,tokenizer)

    def get_char2tok_spanlis_one_doc(self,doc,tokenizer):
        res=[]
        for seq in doc.seqlis:

            token_span = tokenizer.encode_plus(seq, return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]
            # token span中保存的是每个token对应的原文中的[chaidst,charided] 
            char_num = None
            for tok_ind in range(len(token_span) - 1, -1, -1):
                if token_span[tok_ind][1] != 0:
                    char_num = token_span[tok_ind][1]
                    break
            # print(char_num)
            # 建立文本与tokens之间的对应关系
            # 经过实验发现对于原文中的多个空格会被识别成为 一个空格
            char2tok_span = [[-1, -1] for _ in range(char_num)] # [-1, -1] is whitespace
            for tok_ind, char_sp in enumerate(token_span):
                for char_ind in range(char_sp[0], char_sp[1]):
                    tok_sp = char2tok_span[char_ind]
                    # 因为char to tok 也可能出现1对多的情况，比如韩文。所以char_span的pos1以第一个tok_ind为准，pos2以最后一个tok_ind为准
                    if tok_sp[0] == -1:
                        tok_sp[0] = tok_ind
                    tok_sp[1] = tok_ind + 1 #这里在英文里一定是一个char不会出现多个token的情况 所以0就表示对应的tok
            
            res.append(char2tok_span)
        
        return res
            
    #########################################################
    def __getitem__(self, idx):
        """sample data to get batch"""
        seq = self.dataset[idx][0]
        # link = self.dataset[idx][1]
        ele_labels=self.dataset[idx][1]
        qs_role_labels=self.dataset[idx][2]
        return [seq,ele_labels,qs_role_labels]

    def __len__(self):
        """get dataset size"""
        return len(self.dataset)
    def collate_fn(self, batch):
        # 生成一个batch的数据 一个batch就是一个doc
        seqs = [x[0] for x in batch]
        # links= [x[1] for x in batch]
        ele_labels = [x[1] for x in batch]
        qs_role_labels = [x[2] for x in batch]
        
        batch_len = len(seqs)
        max_len = max([len(s) for s in seqs])
        batch_data=[[0 for i in range(max_len)]for j in range(batch_len)]
        for j in range(batch_len):
            cur_len = len(seqs[j])
            batch_data[j][:cur_len] = seqs[j]

        batch_ele_labels=[[0 for i in range(max_len)]for j in range(batch_len)]
        for j in range(batch_len):
            cur_len = len(seqs[j])
            batch_ele_labels[j][:cur_len] = ele_labels[j]

        batch_qs_role_labels=[[0 for i in range(max_len)]for j in range(batch_len)]
        for j in range(batch_len):
            cur_len = len(seqs[j])
            batch_qs_role_labels[j][:cur_len] = qs_role_labels[j]

        batch_seq_mask = [[0 for i in range(max_len)]for j in range(batch_len)]
        for j in range(batch_len):
            cur_len = len(seqs[j])
            batch_seq_mask[j][:cur_len] = [1 for _ in range(cur_len)]
        
        batch_data = torch.tensor(batch_data, dtype=torch.long)
        batch_ele_labels = torch.tensor(batch_ele_labels,dtype=torch.long)
        batch_qs_role_labels = torch.tensor(batch_qs_role_labels,dtype=torch.long)
        batch_seq_mask = torch.tensor(batch_seq_mask,dtype=torch.bool)

        batch_data=batch_data.to(self.device)
        batch_ele_labels=batch_ele_labels.to(self.device)
        batch_qs_role_labels=batch_qs_role_labels.to(self.device)
        batch_seq_mask=batch_seq_mask.to(self.device)
        return {
            'datas':batch_data,
            'ele_labels':batch_ele_labels,
            'role_labels':batch_qs_role_labels,
            'seq_mask':batch_seq_mask
        }
    
if __name__ == '__main__':
    processor = Processor(config)
    processor.process()
    # train_dataset = NERDataset(processor.train_oridocdic,processor.train_docid2docname, config)
    dev_dataset = EleDataset(processor.vail_oridocdic,processor.vail_docid2docname, config)
    # for i in dev_dataset:
    #     print(i)