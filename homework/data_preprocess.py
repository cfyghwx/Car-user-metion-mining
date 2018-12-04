# -*- coding: utf-8 -*-

import os
import re
import jieba
import _pickle as pickle
import random
import pandas as pd
from itertools import islice
from constant import config

stopwords=set()
with open('data/stopwords.txt','r',encoding='utf-8') as fr_words:
    for line in fr_words:
        line=line.strip()
        stopwords.add(line)

def del_stopwords(cont_split):
    new_cont_split=[x for x in cont_split if x not in stopwords]
    return new_cont_split

def data_process(in_path,out_path):
    subject_labels=config.subject_labels
    subject2id=config.subject2id
    id2subject=config.id2subject
    data = pd.read_csv(in_path)
    content_id=data['content_id']
    content=data['content']
    subject=data['subject']
    sentiment_value=data['sentiment_value']
    # sentiment_word=data['sentiment_word']
    content_seg=[]
    for sentence in content:
        seg=jieba.cut(sentence.strip())
        seg=del_stopwords(seg)
        content_seg.append(' '.join(seg))
    new_cont=pd.Series(content_seg)
    subject_id=[]
    for sub in subject:
        subject_id.append(subject2id[sub])
    new_subject=pd.Series(subject_id)
    result = pd.DataFrame({'content_id':content_id,'content':new_cont,'subject':subject,'sentiment_value':sentiment_value})
    result.to_csv(out_path,header=True,index=False)

def seg_train_val(in_path,train_path,val_path):
    with open(in_path,'r',encoding='utf-8') as fr,open(train_path,'w',encoding='utf-8') as fw_train,open(val_path,'w',encoding='utf-8') as fw_val:
        cont=fr.readlines()
        fw_train.write(cont[0])
        fw_val.write(cont[0])
        for line in cont[1:]:
            if random.random()<=0.9:
                fw_train.write(line)
            else:
                fw_val.write(line)


if __name__=='__main__':
    data_process('data/train.csv','data/data.csv')
    seg_train_val('data/data.csv','data/new_train.csv','data/new_val.csv')
    print('over')

