import sys
import os
sys.path.append('./code/')
import torch
import torch.nn as nn
import random
import numpy as np
import torch.nn.functional as F
'''
手动实现transformers
'''
def seed_everything(seed=1029):
    '''
    设置整个开发环境的seed
    :param seed:
    :return:
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True # 这里设置为true是为了保证算法的可复现性
seed_everything()
batch_size=2
voc_len=8
model_dim=10


src_len=torch.randint(2,5,(batch_size,))
tgt_len=torch.randint(2,5,(batch_size,))
max_src_len=max(src_len)
max_tgt_len=max(tgt_len)
max_position_len=max(max_src_len,max_tgt_len)

#### 填充0
src_seq=[F.pad(torch.randint(1,voc_len,(L,)),(0,max_src_len-L)).unsqueeze(0) for L in src_len]
tgt_seq=[F.pad(torch.randint(1,voc_len,(L,)),(0,max_tgt_len-L)).unsqueeze(0) for L in tgt_len]


src_tensor=torch.cat(src_seq,dim=0)
tgt_tensor=torch.cat(tgt_seq,dim=0)
# print(src_tensor)
# print(tgt_tensor)
src_embedding=nn.Embedding(voc_len+1,model_dim)
tgt_embedding=nn.Embedding(voc_len+1,model_dim)

src_emb=src_embedding(src_tensor)
tgt_emb=tgt_embedding(tgt_tensor)
# print(src_emb.size())
# print(tgt_emb.size())

# 构造position embedding
# 矩阵大小是(max_pos_len,model_dim)
pos_mat=torch.arange(max_position_len).reshape((-1,1))
i_mat=torch.pow(10000,torch.arange(0,model_dim,2).reshape((1,-1))/model_dim)
pe_embedding=torch.zeros(max_position_len,model_dim)
pe_embedding[:,0::2]=torch.sin(pos_mat/i_mat)
pe_embedding[:,1::2]=torch.cos(pos_mat/i_mat)
# print(pe_embedding.size())

pe_emb_table=nn.Embedding(max_position_len,model_dim)
pe_emb_table.weight=nn.Parameter(pe_embedding,requires_grad=False)

src_pos_tensor=torch.cat([torch.arange(max_src_len).unsqueeze(0) for _ in src_seq],dim=0)
tgt_pos_tensor=torch.cat([torch.arange(max_tgt_len).unsqueeze(0) for _ in tgt_seq],dim=0)
# print(src_pos_tensor,tgt_pos_tensor)
src_pos_emb=pe_emb_table(src_pos_tensor)
tgt_pos_emb=pe_emb_table(tgt_pos_tensor)
# print(src_pos_emb.size())
# print(tgt_pos_emb.size())
### 构建mask矩阵
vaild_encode_pos=torch.cat([F.pad(torch.ones(L),(0,max_src_len-L)).unsqueeze(0) for L in src_len],dim=0).unsqueeze(-1)
# print(vaild_encode_pos)
vaild_encode_pos_matrix=torch.bmm(vaild_encode_pos,vaild_encode_pos.transpose(1,2))
invaild_encode_pos_matrix = 1-vaild_encode_pos_matrix
mask_encode_self_attention=invaild_encode_pos_matrix.to(torch.bool)
# 如果得到q*k的值，那么可以使用masked_fill方法将True的部分赋值为-inf那么softmax后的值就变成了0不会产生影响
# position 编码使用sin和cos的目的是为了增加位置编码的泛化能力，因为它们是周期函数，所以可以通过周期变化得到任意长度的position编码
vaild_decode_pos=torch.cat([F.pad(torch.ones(L),(0,max_tgt_len-L)).unsqueeze(0) for L in tgt_len],dim=0).unsqueeze(-1)
vaild_crosscode_pos_matrix=torch.bmm(vaild_decode_pos,vaild_encode_pos.transpose(1,2))

invaild_crosscode_pos_matrix= 1-vaild_crosscode_pos_matrix
mask_crosscode_pos_matrix=invaild_crosscode_pos_matrix.to(torch.bool)

vaild_decode_pos=torch.cat([F.pad(torch.ones(L),(0,max_tgt_len-L)).unsqueeze(0) for L in tgt_len],dim=0).unsqueeze(-1)

vaild_decode_tril_matrix=torch.cat([F.pad(torch.tril(torch.ones((L,L))),(0,max_tgt_len-L,0,max_tgt_len-L)).unsqueeze(0) for L in tgt_len],dim=0)
invaild_decode_tril_matrix = 1 - vaild_decode_tril_matrix
mask_decode_tril_matrix=invaild_decode_tril_matrix.to(torch.bool)
# print(mask_decode_tril_matrix)

# 构建self-attention
def scale_dot_product_attention(Q,K,V,attn_mask):
    score = torch.bmm(Q,K.transpose(-2,-1))/torch.sqrt(model_dim)
    masked_score=score.masked_fill(attn_mask,-1e9)
    prob = F.softmax(masked_score,-1)
    context=torch.bmm(prob,V)
    return context
