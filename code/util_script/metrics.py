import os
import sys
sys.path.append('./code/')
import config_script.config as config
import logging

from sklearn.metrics import recall_score,precision_score,f1_score,confusion_matrix,roc_curve,accuracy_score

def p_score_1(y_true,y_pred):
    return precision_score(y_true,y_pred,labels=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],average='macro')

def r_score_1(y_true,y_pred):
    return recall_score(y_true,y_pred,labels=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],average='macro')


def f1_score_ele(y_true,y_pred):
    return f1_score(y_true,y_pred,labels=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],average='macro')

def f1_score_qs_role(y_true,y_pred):
    return f1_score(y_true,y_pred,labels=[1,2,3,4,5,6],average='macro')

def f1_score_3(y_true,y_pred):
    return f1_score(y_true,y_pred,labels=[1,2,3],average='macro')

def f1_score_2(y_true, y_pred, mode='dev'):
    """Compute the F1 score.
    The F1 score can be interpreted as a weighted average of the precision and
    recall, where an F1 score reaches its best value at 1 and worst score at 0.
    The relative contribution of precision and recall to the F1 score are
    equal. The formula for the F1 score is::
        F1 = 2 * (precision * recall) / (precision + recall)
    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a tagger.
    Returns:
        score : float.
    Example:
        y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        f1_score(y_true, y_pred)
        0.50
    """
    true_entities = set(y_true)
    pred_entities = set(y_pred)
    nb_correct = len(true_entities & pred_entities)
    nb_pred = len(pred_entities)
    nb_true = len(true_entities)

    p = nb_correct / nb_pred if nb_pred > 0 else 0
    r = nb_correct / nb_true if nb_true > 0 else 0
    score = 2 * p * r / (p + r) if p + r > 0 else 0
    if mode == 'dev':
        return score
    else:
        f_score = {}
        for label in config.labels:
            true_entities_label = set()
            pred_entities_label = set()
            for t in true_entities:
                if t[0] == label:
                    true_entities_label.add(t)
            for p in pred_entities:
                if p[0] == label:
                    pred_entities_label.add(p)
            nb_correct_label = len(true_entities_label & pred_entities_label)
            nb_pred_label = len(pred_entities_label)
            nb_true_label = len(true_entities_label)

            p_label = nb_correct_label / nb_pred_label if nb_pred_label > 0 else 0
            r_label = nb_correct_label / nb_true_label if nb_true_label > 0 else 0
            score_label = 2 * p_label * r_label / (p_label + r_label) if p_label + r_label > 0 else 0
            f_score[label] = score_label
        return f_score, score



'''
CRF 的F1值的计算
'''
def get_entities(seq):
    """
    Gets entities from sequence.
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        seq = ['B-PER', 'I-PER', 'O', 'B-LOC'] 这里仅使用B I O 
        get_entities(seq)
        [('PER', 0, 1), ('LOC', 3, 3)]
    """
    # for nested list
    if any(isinstance(s, list) for s in seq):
        seq = [item for sublist in seq for item in sublist + ['O']]
        #做了展开
    #******* 下面这个操作是为了去掉以I开始的错误标签****************
    prev_tag='O'
    for i in range(len(seq)):
        if(seq[i].split('-')[0]=='I' and prev_tag=='O'):
            seq[i]='O'
        prev_tag=seq[i].split('-')[0]
    #************************************************************
    prev_tag = 'O'
    prev_type = ''
    begin_offset = 0
    chunks = []
    for i, chunk in enumerate(seq + ['O']):
        tag = chunk.split('-')[0]
        type_ = chunk.split('-')[-1]
        #这里仅考虑B 是不是会好一点？
        # 虽然有提升 但是效果依旧很差 不知道为啥
        # if(tag=='B'):
        #     chunks.append((type_,i,i))
        if end_of_chunk(prev_tag, tag, prev_type, type_):
            chunks.append((prev_type, begin_offset, i - 1))
        if start_of_chunk(prev_tag, tag, prev_type, type_):
            begin_offset = i
        prev_tag = tag
        prev_type = type_

    return chunks


def end_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk ended between the previous and current word.
    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.
    Returns:
        chunk_end: boolean.
    """
    chunk_end = False

    # if prev_tag == 'S':
    #     chunk_end = True
    # pred_label中可能出现这种情形
    if prev_tag == 'B' and tag == 'B':
        chunk_end = True
    # if prev_tag == 'B' and tag == 'S':
    #     chunk_end = True
    if prev_tag == 'B' and tag == 'O':
        chunk_end = True
    if prev_tag == 'I' and tag == 'B':
        chunk_end = True
    # if prev_tag == 'I' and tag == 'S':
    #     chunk_end = True
    if prev_tag == 'I' and tag == 'O':
        chunk_end = True

    if prev_tag != 'O' and prev_tag != '.' and prev_type != type_:
        # B I
        chunk_end = True
    # 

    return chunk_end


def start_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk started between the previous and current word.
    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.
    Returns:
        chunk_start: boolean.
    """
    chunk_start = False

    if tag == 'B':
        chunk_start = True
    # if tag == 'S':
    #     chunk_start = True

    # if prev_tag == 'S' and tag == 'I':
    #     chunk_start = True
    if prev_tag == 'O' and tag == 'I':
        #这里相当于考虑了以I开始的数据
        chunk_start = True

    if tag != 'O' and tag != '.' and prev_type != type_:
        # tag = B I  这里把I开始也看作是一个实体了
        chunk_start = True

    return chunk_start


def f1_score(y_true, y_pred, mode='dev'):
    """Compute the F1 score.
    Macro-average
    The F1 score can be interpreted as a weighted average of the precision and
    recall, where an F1 score reaches its best value at 1 and worst score at 0.
    The relative contribution of precision and recall to the F1 score are
    equal. The formula for the F1 score is::
        F1 = 2 * (precision * recall) / (precision + recall)
    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a tagger.
    Returns:
        score : float.
    Example:
        y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        f1_score(y_true, y_pred)
        0.50
    """
    id2ele={v:k for k,v in config.spatial_qsomvrole_label2id.items()}
    # print(y_true)
    y_true=[id2ele[idx] for idx in y_true]
    y_pred=[id2ele[idx] for idx in y_pred]
    #这里计算是没有问题的
    true_entities = set(get_entities(y_true))
    pred_entities = set(get_entities(y_pred))
    nb_correct = len(true_entities & pred_entities)
    nb_pred = len(pred_entities)
    nb_true = len(true_entities)

    p = nb_correct / nb_pred if nb_pred > 0 else 0
    r = nb_correct / nb_true if nb_true > 0 else 0
    score = 2 * p * r / (p + r) if p + r > 0 else 0
    if mode == 'dev':
        return score
    else:
        f_score = {}
        for label in config.labels:
            true_entities_label = set()
            pred_entities_label = set()
            for t in true_entities:
                if t[0] == label:
                    true_entities_label.add(t)
            for p in pred_entities:
                if p[0] == label:
                    pred_entities_label.add(p)
            nb_correct_label = len(true_entities_label & pred_entities_label)
            nb_pred_label = len(pred_entities_label)
            nb_true_label = len(true_entities_label)

            p_label = nb_correct_label / nb_pred_label if nb_pred_label > 0 else 0
            r_label = nb_correct_label / nb_true_label if nb_true_label > 0 else 0
            score_label = 2 * p_label * r_label / (p_label + r_label) if p_label + r_label > 0 else 0
            f_score[label] = score_label
        return f_score, score


# def bad_case(y_true, y_pred, data):
#     if not os.path.exists(config.case_dir):
#         os.system(r"touch {}".format(config.case_dir))  # 调用系统命令行来创建文件
#     output = open(config.case_dir, 'w')
#     for idx, (t, p) in enumerate(zip(y_true, y_pred)):
#         if t == p:
#             continue
#         else:
#             output.write("bad case " + str(idx) + ": \n")
#             output.write("sentence: " + str(data[idx]) + "\n")
#             output.write("golden label: " + str(t) + "\n")
#             output.write("model pred: " + str(p) + "\n")
#     logging.info("--------Bad Cases reserved !--------")