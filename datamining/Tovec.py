# coding=utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import jieba
from gensim.models import Doc2Vec
from gensim.models import Word2Vec
from TaggedLineSentence import TaggedLineSentence


#train doc2vec model
def get_model2(senlist, i=1, j=0.025,vsize=100):
    tls = TaggedLineSentence()
    tagged_stc = tls.to_array(senlist)
    model = Doc2Vec(min_count=1, vector_size=vsize, window=8, sample=1e-4, negative=6, dm=0, workers=3,alpha=0.115)
    model.build_vocab(tagged_stc)
    model.train(tls.perm(),
                total_examples=model.corpus_count,
                epochs=360)
    return model



def tovec(i,j,vsize):
    f = open('D:\\南京大学\\研一\\上课材料+作业\\作业\\data mining\\大作业\\train.csv', encoding='utf-8')
    # data为pandas对象，类似于表格的样子
    data = pd.read_csv(f)
    # print(data)
    npdata = data.values
    npdata = np.array(npdata)
    sentences = npdata[:, 1]
    sen_count = len(sentences)
    sentences_list = []
    for sen in sentences:
        sen_fc = jieba.cut(sen, cut_all=False)
        sen_fc = ' '.join(sen_fc).replace(',', '').replace('，', '')
        sen_fc = sen_fc.split(' ')
        sentences_list.append(sen_fc)
        # print(sen_fc)
    model = get_model2(sentences_list,i,j,vsize)
    ac_matrix = np.zeros((sen_count, vsize))
    for i in range(sen_count):
        ac_matrix[i] = model.docvecs[i]
    return ac_matrix


def tovec_extract(i,j,vsize):
    f = open('D:\\南京大学\\研一\\上课材料+作业\\作业\\data mining\\大作业\\new_train.csv', encoding='utf-8')
    # data为pandas对象，类似于表格的样子
    data = pd.read_csv(f)
    # print(data)
    npdata = data.values
    npdata = np.array(npdata)
    sentences = npdata[:, 1]
    sen_count = len(sentences)
    sentences_list = []
    for sen in sentences:
        # sen_fc = jieba.cut(sen, cut_all=False)
        # sen_fc = ' '.join(sen_fc).replace(',', '').replace('，', '')
        sen = sen.split(' ')
        # print(sen)
        sentences_list.append(sen)
        # print(sen_fc)
    model = get_model2(sentences_list,i,j,vsize)
    ac_matrix = np.zeros((sen_count, vsize))
    for i in range(sen_count):
        ac_matrix[i] = model.docvecs[i]


    return ac_matrix

def tovec_csv_udefsize(i,j,vsize):
    f = open('D:\\南京大学\\研一\\上课材料+作业\\作业\\data mining\\大作业\\new_train.csv', encoding='utf-8')
    # data为pandas对象，类似于表格的样子
    data = pd.read_csv(f)
    # print(data)
    npdata = data.values
    npdata = np.array(npdata)
    sentences = npdata[:, 1]
    sen_count = len(sentences)
    sentences_list = []
    for sen in sentences:
        sen = sen.split(' ')
        sentences_list.append(sen)
    model = get_model2(sentences_list, i, j, vsize)
    ac_matrix = np.zeros((sen_count, vsize))
    for i in range(sen_count):
        ac_matrix[i] = model.docvecs[i]
    # print(ac_matrix)
    np.savetxt('new.csv', ac_matrix, delimiter = ',')


