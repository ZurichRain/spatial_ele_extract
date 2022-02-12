#这个是数据预处理代码 将数据打包之后的原始数据 和 句子级别的原始数据
#目前的理解是对于每一个spatial_role 都建立一个二分类器 比如：[1,0,0] 对每个角色做区分可能效果会更好
# 目前先处理step1 的数据情况 就是给每个token 打标签 采用BIO
# 首先要对每一句话进行分割 并将原来的spatial element 的ids 映射为{seqstid,seqedid}
# 对于跨句子的span 目前遇到再想办法 

# [start,end) 其中end是开区间 
import os
import sys
# print(os.getcwd())
sys.path.append('./code/')

import numpy as np
import pandas as pd
import xml.dom.minidom as xmd
import logging
import config_script.config as config
# import torch
# from torch import nn
# from transformers import pipeline

from util_script.mata_data_calss import *

logging.basicConfig(level=logging.DEBUG)


class Processor(object):
    def __init__(self, config):
        # config.train_CP_dir config.train_ANC_dir config.train_RFC_dir
        self.train_data_dir = config.train_CP_dir
        self.vail_data_dir = config.vail_data_dir
        self.test_data_dir= config.test_data_dir
        self.config = config
    def process(self):
        """
        process train and test data  先把train的问题解决了
        """
        self.train_oridocdic,self.train_docid2docname=self.preprocess(self.train_data_dir) #这里仅仅是将数据保存在对象里 以后可以写到文件中
        self.vail_oridocdic,self.vail_docid2docname=self.preprocess(self.vail_data_dir)
        self.test_oridocdic,self.test_docid2docname=self.preprocess(self.test_data_dir)
        # import pandas as pd
        
        
        # for k,v in self.train_oridocdic.items():
        #     data = {}
        #     for idx,seq in enumerate(v.seqlis):
        #         data[str(idx)]='['+seq+']'
        #     data_df = pd.DataFrame(data,index=[0])
        #     data_df.to_csv('/data2/fwang/baseline/data/my_process_data/train/'+self.train_docid2docname[k].split('/')[-1][:-4]+'_seqs.csv')
        #         # print(str(idx)+seq)
        #         # f.writelines('***'+str(idx)+'***\n['+seq+']\n')
        #     # with open('/data2/fwang/baseline/data/my_process_data/train/'+self.train_docid2docname[k].split('/')[-1][:-4]+'_seqs.csv','w') as f:
        #         # print(len(v.seqlis))
                

        # for k,v in self.vail_oridocdic.items():

        #     data = {}
        #     for idx,seq in enumerate(v.seqlis):
        #         data[str(idx)]='['+seq+']'
        #     data_df = pd.DataFrame(data,index=[0])
        #     data_df.to_csv('/data2/fwang/baseline/data/my_process_data/vail/'+self.vail_docid2docname[k].split('/')[-1][:-4]+'_seqs.csv')
        #     # with open('/data2/fwang/baseline/data/my_process_data/vail/'+self.vail_docid2docname[k].split('/')[-1][:-4]+'_seqs.csv','w') as f:
        #     #     for idx,seq in enumerate(v.seqlis):
        #     #         # print(str(idx)+seq)
        #     #         f.writelines('***'+str(idx)+'***\n['+seq+']\n')

        # for k,v in self.test_oridocdic.items():
        #     data = {}
        #     for idx,seq in enumerate(v.seqlis):
        #         data[str(idx)]='['+seq+']'
        #     data_df = pd.DataFrame(data,index=[0])
        #     data_df.to_csv('/data2/fwang/baseline/data/my_process_data/test/'+self.test_docid2docname[k].split('/')[-1][:-4]+'_seqs.csv')
        #     # with open('/data2/fwang/baseline/data/my_process_data/test/'+self.test_docid2docname[k].split('/')[-1][:-4]+'_seqs.csv','w') as f:
        #     #     for idx,seq in enumerate(v.seqlis):
        #     #         # print(str(idx)+seq)
        #     #         f.writelines('***'+str(idx)+'***\n['+seq+']\n')
        # print(len(self.train_oridocdic)) 55 个感觉上应该没有问题
        # print(len(self.vail_oridocdic)) 23个感觉上也没有啥问题
        # for file_name in self.config.files:
        #     self.preprocess(file_name)
    def parse_data_from_xml(self,xml_file):
        id2obj=dict()
        res_doc=ori_data() #将数据转化为一个对象方便之后的处理
        # print(xml_file)
        dom=xmd.parse(xml_file)
        root = dom.documentElement
        texts = root.getElementsByTagName('TEXT')
        for text in texts:  
            for child in text.childNodes:
                #如果有多段话 就是用空格隔开 好像是每个doc只有一个text
                res_doc.text+=child.data
                res_doc.text+=' '


        places = dom.getElementsByTagName('PLACE')
        for p in places:
            aplace=place()
            for k,v in p.attributes.items():
                curk=k.lower()#这里是先对属性变为小写
                aplace.update_attr(curk,v)
            idv=p.getAttribute('id')#有可能出现id不存在的情况（我手动去除了）
            id2obj[idv]=aplace


            res_doc.place_lis.append(aplace)


        paths = dom.getElementsByTagName('PATH')
        for p in paths:
            apath=path()
            for k,v in p.attributes.items():
                curk=k.lower()
                apath.update_attr(curk,v)

            idv=p.getAttribute('id')
            id2obj[idv]=apath

            res_doc.path_lis.append(apath)


        spatial_entitys = dom.getElementsByTagName('SPATIAL_ENTITY')
        for p in spatial_entitys:
            aspatial_entity=spatial_entity()
            for k,v in p.attributes.items():
                curk=k.lower()
                aspatial_entity.update_attr(curk,v)

            idv=p.getAttribute('id')
            id2obj[idv]=aspatial_entity
            
            res_doc.spatial_entity_lis.append(aspatial_entity)


        nonmotion_events = dom.getElementsByTagName('NONMOTION_EVENT')
        for p in nonmotion_events:
            anonmotion_event=nonmotion_event()
            for k,v in p.attributes.items():
                curk=k.lower()
                anonmotion_event.update_attr(curk,v)

            idv=p.getAttribute('id')
            id2obj[idv]=anonmotion_event

            res_doc.nonmotion_event_lis.append(anonmotion_event)

        motions = dom.getElementsByTagName('MOTION')
        for p in motions:
            amotion=motion()
            for k,v in p.attributes.items():
                curk=k.lower()
                amotion.update_attr(curk,v)

            idv=p.getAttribute('id')
            id2obj[idv]=amotion

            res_doc.motion_lis.append(amotion)

        spatial_signals = dom.getElementsByTagName('SPATIAL_SIGNAL')
        for p in spatial_signals:
            aspatial_signal=spatial_signal()
            for k,v in p.attributes.items():
                curk=k.lower()
                aspatial_signal.update_attr(curk,v)

            idv=p.getAttribute('id')
            id2obj[idv]=aspatial_signal
            res_doc.spatial_signal_lis.append(aspatial_signal)

        motion_signals = dom.getElementsByTagName('MOTION_SIGNAL')
        for p in motion_signals:
            amotion_signal=motion_signal()
            for k,v in p.attributes.items():
                curk=k.lower()
                amotion_signal.update_attr(curk,v)

            idv=p.getAttribute('id')
            id2obj[idv]=amotion_signal
            res_doc.motion_signal_lis.append(amotion_signal)

        measures = dom.getElementsByTagName('MEASURE')
        for p in measures:
            ameasure=measure()
            for k,v in p.attributes.items():
                curk=k.lower()
                ameasure.update_attr(curk,v)

            idv=p.getAttribute('id')
            id2obj[idv]=ameasure
            res_doc.measure_lis.append(ameasure)

        qslinks = dom.getElementsByTagName('QSLINK')
        for p in qslinks:
            aqslink=qslink()
            for k,v in p.attributes.items():
                curk=k.lower()
                aqslink.update_attr(curk,v)

            idv=p.getAttribute('id')
            id2obj[idv]=aqslink
            res_doc.qslink_lis.append(aqslink)

        olinks = dom.getElementsByTagName('OLINK')
        for p in olinks:
            aolink=olink()
            for k,v in p.attributes.items():
                curk=k.lower()
                aolink.update_attr(curk,v)

            idv=p.getAttribute('id')
            id2obj[idv]=aolink

            res_doc.olink_lis.append(aolink)
        
        movelinks = dom.getElementsByTagName('MOVELINK')
        for p in movelinks:
            amovelink=movelink()
            for k,v in p.attributes.items():
                curk=k.lower()
                amovelink.update_attr(curk,v)

            idv=p.getAttribute('id')
            id2obj[idv]=amovelink
            res_doc.movelink_lis.append(amovelink)

        measurelinks = dom.getElementsByTagName('MEASURELINK')
        for p in measurelinks:
            ameasurelink=measurelink()
            for k,v in p.attributes.items():
                curk=k.lower()
                ameasurelink.update_attr(curk,v)

            idv=p.getAttribute('id')
            id2obj[idv]=ameasurelink
            res_doc.measurelink_lis.append(ameasurelink)

        metalinks = dom.getElementsByTagName('METALINK')
        for p in metalinks:
            ametalink=metalink()
            for k,v in p.attributes.items():
                curk=k.lower()
                ametalink.update_attr(curk,v)

            idv=p.getAttribute('id')
            id2obj[idv]=ametalink
            res_doc.metalink_lis.append(ametalink)

        res_doc.id2obj=id2obj
        
        # docid2id2obj[docid]=id2obj
        # print(res_doc.text)
        # print(res_doc.qslink_lis)
        # print(id2obj)
        return res_doc
    def get_stedids_element_relation(self,doc):
        #用来建立空间元素下标和空间元素对象之间的关系 方便寻找在某个范围内的所有元素 为之后计算元素在句子中的位置做铺垫
        #原文中的开始位置是包含 结束位置是不包含
        #这里有可能会出现一个token扮演不同的空间元素 目前的处理方法是先将它选择其中之一进行预测
        #muti 空间元素的数量 训练集有 8个 验证集有 2个
        stedids2element=dict()
        element2stedids=dict()
        stedele_lis=[]
        for lin in doc.spatial_entity_lis:

            if(int(lin.start)==-1):
                #排除掉没有原文下标的句子元素
                continue
            # if(stedids2element.get((int(lin.start),int(lin.end)))):
            #     stedids2element[(int(lin.start),int(lin.end))].append(lin)
            # else:
            #     stedids2element[(int(lin.start),int(lin.end))]=[lin]
            # if(stedids2element.get((int(lin.start),int(lin.end)))):
            #     multi_ele+=1
            stedids2element[(int(lin.start),int(lin.end))]=lin
            element2stedids[lin]=(int(lin.start),int(lin.end))
            stedele_lis.append((int(lin.start),int(lin.end)))

        for lin in doc.nonmotion_event_lis:

            if(int(lin.start)==-1):
                continue
            # if(stedids2element.get((int(lin.start),int(lin.end)))):
            #     multi_ele+=1
            stedids2element[(int(lin.start),int(lin.end))]=lin
            element2stedids[lin]=(int(lin.start),int(lin.end))
            stedele_lis.append((int(lin.start),int(lin.end)))

        for lin in doc.motion_lis:

            if(int(lin.start)==-1):
                continue
            # if(stedids2element.get((int(lin.start),int(lin.end)))):
            #     multi_ele+=1
            stedids2element[(int(lin.start),int(lin.end))]=lin
            element2stedids[lin]=(int(lin.start),int(lin.end))
            stedele_lis.append((int(lin.start),int(lin.end)))

        for lin in doc.spatial_signal_lis:

            if(int(lin.start)==-1):
                continue
            # if(stedids2element.get((int(lin.start),int(lin.end)))):
            #     multi_ele+=1
            stedids2element[(int(lin.start),int(lin.end))]=lin
            element2stedids[lin]=(int(lin.start),int(lin.end))
            
            stedele_lis.append((int(lin.start),int(lin.end)))

        for lin in doc.motion_signal_lis:

            if(int(lin.start)==-1):
                continue
            # if(stedids2element.get((int(lin.start),int(lin.end)))):
            #     multi_ele+=1
            stedids2element[(int(lin.start),int(lin.end))]=lin
            element2stedids[lin]=(int(lin.start),int(lin.end))
            stedele_lis.append((int(lin.start),int(lin.end)))

        for lin in doc.measure_lis:

            if(int(lin.start)==-1):
                continue
            # if(stedids2element.get((int(lin.start),int(lin.end)))):
            #     multi_ele+=1
            stedids2element[(int(lin.start),int(lin.end))]=lin
            element2stedids[lin]=(int(lin.start),int(lin.end))
            stedele_lis.append((int(lin.start),int(lin.end)))
        
        for lin in doc.place_lis:
            
            if(int(lin.start)==-1):
                continue
            # if(stedids2element.get((int(lin.start),int(lin.end)))):
            #     multi_ele+=1
            stedids2element[(int(lin.start),int(lin.end))]=lin
            element2stedids[lin]=(int(lin.start),int(lin.end))
            stedele_lis.append((int(lin.start),int(lin.end)))

        for lin in doc.path_lis:

            if(int(lin.start)==-1):
                continue
            # if(stedids2element.get((int(lin.start),int(lin.end)))):
            #     multi_ele+=1
            stedids2element[(int(lin.start),int(lin.end))]=lin
            element2stedids[lin]=(int(lin.start),int(lin.end))
            stedele_lis.append((int(lin.start),int(lin.end)))

        stedele_lis=sorted(stedele_lis)
        
        return stedids2element,element2stedids,stedele_lis
    
    def get_spatial_seqstedids(self,seqidslis,stedids2element,stedele_lis,seqid):
        one_seq_eles=[]
        for i in stedele_lis:
            if(i[1]>seqidslis[0] and i[0]>=seqidslis[0] and i[1]<=seqidslis[-1]+1):
                ele=stedids2element[i]
                ele.add_attr('seqstid',(seqid,int(ele.start)-seqidslis[0]))
                ele.add_attr('seqedid',(seqid,int(ele.end)-seqidslis[0]))
                one_seq_eles.append(ele)
        return one_seq_eles
    def check(self,alltext,cid,stchar,endsig):
        c=cid
        if(alltext[c]=='S' or alltext[c]=='A'):
            #这里特判了两个地名 他们虽然后面是. 但是他们不是一句话的终结
            return False
        while(c<len(alltext)):
            if(alltext[c]==' ' or alltext[c] =='\n' or alltext[c] in endsig):
                c+=1
            elif(alltext[c] in stchar):
                return True
            elif(alltext[c] not in stchar):
                return False

    def get_doc_seq_lis(self,oridocdic,docid2docname):
        #判断句子这里情况比较多 比较复杂 所以我这里是给了一个规则
        #有些文档里就一句话
        endsig=['.','!','?']
        stchar=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
        # 一个句子中空格是允许出现的 但是\n不允许出现
        for k,v in oridocdic.items():
            #得到一个文档中的所有element的下标和对象之间的关系
            # print(docid2docname[k])
            stedids2element,element2stedids,stedele_lis=self.get_stedids_element_relation(v)
            curseqlis=[]
            curseqoriids=[]
            curseqtxt=''
            seq_ori_charids_lis=[]#保存当前句子在原始文档中的字符位置
            seq_ele_lis=[]
            alltext=v.text
            seqid=0
            for cid in range(len(alltext)):
                if((alltext[cid-1] in endsig and alltext[cid]==' ') and self.check(alltext,cid,stchar,endsig)):
                    continue
                if(alltext[cid] in endsig and self.check(alltext,cid+1,stchar,endsig)):
                    #如果当前句子以结束词结尾 并且当前句子的下一个开始时下一个句子 那么就处理当前句子 
                    curseqtxt+=alltext[cid]
                    seq_ori_charids_lis.append(cid)
                    curseqlis.append(curseqtxt)
                    curseqoriids.append(seq_ori_charids_lis)
                    one_seq_ele=self.get_spatial_seqstedids(seq_ori_charids_lis,stedids2element,stedele_lis,seqid)
                    seq_ele_lis.append(one_seq_ele)
                    seq_ori_charids_lis=[]
                    curseqtxt=''
                    seqid+=1
                    continue
                #这样处理存在一个问题就是 如果真实数据中存在a.b这样的数据 那么将会切分开来 但是如果考虑后面出现字符这样的情况太复杂 目前这一版就先这样做
                seq_ori_charids_lis.append(cid)
                curseqtxt+=alltext[cid]
            curseqlis.append(curseqtxt)
            curseqoriids.append(seq_ori_charids_lis)
            one_seq_ele=self.get_spatial_seqstedids(seq_ori_charids_lis,stedids2element,stedele_lis,seqid)
            seq_ele_lis.append(one_seq_ele)
            # data = {}
            # for idx,seq_ele in enumerate(seq_ele_lis):
            #     data[str(idx)]=str(seq_ele)
            # data_df = pd.DataFrame(data,index=[0])
            # data_df.to_csv('/data2/fwang/baseline/data/my_process_data/test_eles/'+docid2docname[k].split('/')[-1][:-4]+'_seqeles.csv')
            v.seqlis=curseqlis
            # print(min([len(x) for x in curseqlis]))
            v.seqoriids=curseqoriids

    def preprocess(self, filedir):
        xml_dirs=os.listdir(filedir)
        docid=0
        oridocdic=dict()
        docid2docname=dict()
        for xml_file in xml_dirs:
            if(xml_file[-3:]!='xml'):
                #排除其他元数据文件
                continue
            cur_xml_file=os.path.join(filedir,xml_file)
            curdoc=self.parse_data_from_xml(cur_xml_file)#数据中间会有空格等复杂的情况
            docid2docname[docid]=cur_xml_file
            oridocdic[docid]=curdoc
            docid+=1
        self.get_doc_seq_lis(oridocdic,docid2docname)
        logging.info("--------{} data process DONE!--------".format(filedir))
        return oridocdic,docid2docname

if __name__=='__main__':
    # xml_dir='/home/zurichrain/wf_study/semeval_2015_task8_data_attempt/spatial_data/Traning/CP/'
    # oridocdic,docid2docname=get_ori_doc_info(xml_dir)
    # print(oridocdic[0].id2obj)

    # print(docid2id2obj)
    # docid2docname=dict()
    processor = Processor(config)
    processor.process()
    # print(processor)
    import pickle
    # fn1 = 'train_data_summary.pkl' 
    # with open(fn1, 'wb') as f: # open file with write-mode  
    #     picklestring = pickle.dump(processor.train_oridocdic, f) # serialize and save object
    # fn2 = 'train_data_id_summary.pkl' 
    # with open(fn2, 'wb') as f: # open file with write-mode  
    #     picklestring = pickle.dump(processor.train_docid2docname, f) # serialize and save object
    fn1 = '/data2/fwang/spatial_ele_extract/code/data/data_process_pkl/ori_data_summary_55_14/train_data_summary.pkl'
    fn2 = '/data2/fwang/spatial_ele_extract/code/data/data_process_pkl/ori_data_summary_55_14/train_data_id_summary.pkl'
    fn3 = '/data2/fwang/spatial_ele_extract/code/data/data_process_pkl/ori_data_summary_55_14/vail_data_summary.pkl'
    fn4 = '/data2/fwang/spatial_ele_extract/code/data/data_process_pkl/ori_data_summary_55_14/vail_data_id_summary.pkl'
    fn5 = '/data2/fwang/spatial_ele_extract/code/data/data_process_pkl/ori_data_summary_55_14/test_data_summary.pkl'
    fn6 = '/data2/fwang/spatial_ele_extract/code/data/data_process_pkl/ori_data_summary_55_14/test_data_id_summary.pkl'

    # fn1 = '/data2/fwang/spatial_ele_extract/code/data/data_process_pkl/ori_data_summary_59_16/train_data_summary.pkl'
    # fn2 = '/data2/fwang/spatial_ele_extract/code/data/data_process_pkl/ori_data_summary_59_16/train_data_id_summary.pkl'
    # fn3 = '/data2/fwang/spatial_ele_extract/code/data/data_process_pkl/ori_data_summary_59_16/vail_data_summary.pkl'
    # fn4 = '/data2/fwang/spatial_ele_extract/code/data/data_process_pkl/ori_data_summary_59_16/vail_data_id_summary.pkl'
    # fn5 = '/data2/fwang/spatial_ele_extract/code/data/data_process_pkl/ori_data_summary_59_16/test_data_summary.pkl'
    # fn6 = '/data2/fwang/spatial_ele_extract/code/data/data_process_pkl/ori_data_summary_59_16/test_data_id_summary.pkl'
    # #下面是写入序列化的代码
    # with open(fn1, 'wb') as f:  
    #     picklestring = pickle.dump(processor.train_oridocdic, f)
    # with open(fn2, 'wb') as f:  
    #     picklestring = pickle.dump(processor.train_docid2docname, f)

    # with open(fn3, 'wb') as f:  
    #     picklestring = pickle.dump(processor.vail_oridocdic, f)
    # with open(fn4, 'wb') as f:  
    #     picklestring = pickle.dump(processor.vail_docid2docname, f)

    # with open(fn5, 'wb') as f:  
    #     picklestring = pickle.dump(processor.test_oridocdic, f)
    # with open(fn6, 'wb') as f:  
    #     picklestring = pickle.dump(processor.test_docid2docname, f)
    # 下面是加载的代码
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
    

    # print(train_docid2docname,len(train_docid2docname))
    # print(vail_docid2docname,len(vail_docid2docname))
    # print(test_docid2docname,len(test_docid2docname))
    # with open(fn3, 'wb') as f: # open file with write-mode  
    #     picklestring = pickle.dump(processor.test_oridocdic, f) # serialize and save object
    # with open(fn4, 'wb') as f: # open file with write-mode  
    #     picklestring = pickle.dump(processor.test_docid2docname, f) # serialize and save object
    
    # with open(fn5, 'wb') as f: # open file with write-mode  
    #     picklestring = pickle.dump(processor.vail_oridocdic, f) # serialize and save object
    # with open(fn6, 'wb') as f: # open file with write-mode  
    #     picklestring = pickle.dump(processor.vail_docid2docname, f) # serialize and save object
    # with open(fn1, 'rb') as f:  
    #     train_oridocdic = pickle.load(f)
    # with open(fn2, 'rb') as f:  
    #     train_docid2docname = pickle.load(f)
    # print(train_oridocdic)
    # print(train_docid2docname)
    # print(processor.train_docid2docname[0])
    # doc=processor.train_oridocdic[0]
    # for lin in doc.qslink_lis:
    #     ld=doc.id2obj[lin.landmark]
    #     if(hasattr(ld,'seqstid')):
    #         st=ld.seqstid[1]
    #         ed=ld.seqedid[1]
    #         print(doc.seqlis[ld.seqstid[0]])
    #         print(doc.seqlis[ld.seqstid[0]][st:ed])
    #         print(ld)
    #     print('*'*100)



