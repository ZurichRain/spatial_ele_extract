import sys
import os
sys.path.append('./code/')
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch.nn as nn
import torch
import config_script.config as config


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