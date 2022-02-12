'''
用来统计出现连续多个空格的空间元素有多少？
'''
import sys
import os
sys.path.append('./code/')
from util_script.mata_data_calss import *
from transformers import BertTokenizer,BertModel,BertTokenizerFast,BertPreTrainedModel,get_linear_schedule_with_warmup
import config_script.config as config
import pickle

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

if __name__ == '__main__':
    train_oridocdic,train_docid2docname,vail_oridocdic,vail_docid2docname,test_oridocdic,test_docid2docname=get_three_data(55,14)
    tokenizer = BertTokenizerFast.from_pretrained(config.bert_pathname)
    cnt=0
    for k,v in train_oridocdic.items():
        for elelisname in ele_lis_name:
            for ele in getattr(v,elelisname):
                if('  ' in ele.text):
                    print(train_docid2docname[k])
                    seqids=0
                    for seq in v.seqlis:
                        
                        print(str(seqids)+' ['+seq+']')
                        print(tokenizer.tokenize(seq))
                        print('*'*100)
                        seqids+=1

                    eleseqid=ele.seqstid[0]
                    token_span = tokenizer.encode_plus(v.seqlis[eleseqid], return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]
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
                    st=ele.seqstid[1]
                    ed=ele.seqedid[1]
                    curtok_lis=[]
                    for i in range(st,ed):
                        if(char2tok_span[i][0] not in curtok_lis and char2tok_span[i][0]!=-1 and char2tok_span[i][1]!=-1):
                            curtok_lis.append(char2tok_span[i][0])
                    cnt+=1
                    print(ele)
                    print(tokenizer.tokenize(v.seqlis[eleseqid])[curtok_lis[0]:curtok_lis[-1]+1])
                    print(curtok_lis)
                    exit()
    print(cnt)

    