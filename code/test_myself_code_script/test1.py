import sys
import os
sys.path.append('./code/')
from data_process_script.data_process import *
from transformers import BertTokenizer,BertModel,BertTokenizerFast,BertPreTrainedModel,get_linear_schedule_with_warmup
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained(config.T5base_pathname)
# input_pt = tokenizer('a   b', return_tensors='pt',add_special_tokens=False).input_ids
# token_span = tokenizer.encode_plus('a   b', return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]
seq='stringing   b '
print(tokenizer(seq,add_special_tokens=False).input_ids)
curtok=''
seqtok=[]
seqoridx=[]
st=0
ed=-1
for cidx in range(len(seq)):
    if(seq[cidx] ==' '):
        ed=cidx
        if(st==ed):
            st=cidx+1
            continue
        t5tok=tokenizer.tokenize(curtok,add_special_tokens=False)
        if(len(t5tok)>1):
            cst=st
            ced=st+len(t5tok[0])
            seqtok.append(t5tok[0])
            seqoridx.append((cst,ced))
            cst=ced
            for ct5tok in t5tok[1:]:
                ced=cst+len(ct5tok)
                seqtok.append(ct5tok)
                seqoridx.append((cst,ced))
                cst=ced
        else:
            seqtok.append(t5tok[0])
            seqoridx.append((st,ed))
        curtok=''
        st=cidx+1
    else:
        curtok+=seq[cidx]
print(seqtok)
print(seqoridx)
# for ctok in seq.split(' '):
#     if(len(ctok)>0):
#         t5tok=tokenizer.tokenize(ctok)
#         print(t5tok)

# print(input_pt)
# print(token_span)