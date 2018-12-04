# coding: utf-8

import sys
from collections import Counter

import numpy as np
import pandas as pd
from constant import config
import tensorflow.contrib.keras as kr
import tensorflow as tf

if sys.version_info[0] > 2:
    is_py3 = True
else:
    reload(sys)
    sys.setdefaultencoding("utf-8")
    is_py3 = False


def native_word(word, encoding='utf-8'):
    """如果在python2下面使用python3训练的模型，可考虑调用此函数转化一下字符编码"""
    if not is_py3:
        return word.encode(encoding)
    else:
        return word


def native_content(content):
    if not is_py3:
        return content.decode('utf-8')
    else:
        return content


def open_file(filename, mode='r'):
    """
    常用文件操作，可在python2和python3间切换.
    mode: 'r' or 'w' for read or write
    """
    if is_py3:
        return open(filename, mode, encoding='utf-8', errors='ignore')
    else:
        return open(filename, mode)


def read_file(filename):
    """读取文件数据"""
    data = pd.read_csv(filename)
    contents=data['content'].tolist()
    subjects=data['subject'].tolist()
    labels=data['sentiment_value'].tolist()
    new_contents=[]
    for cont in contents:
        new_contents.append(cont.split(' '))
    return new_contents,subjects,labels


def build_vocab(train_dir, vocab_dir, threshold=5):
    """根据训练集构建词汇表，存储"""
    data_text,data_subject,labels= read_file(train_dir)

    all_data = []  #构建词表
    for content in data_text:
        all_data.extend(content)

    counter = Counter(all_data)
    words=[word for word,cnt in counter.items() if cnt>=threshold]
    # 添加一个 <PAD> 标记符（后续作用：将所有文本pad为同一长度）
    words = ['<PAD>'] + words
    for _ in config.subject_labels:
        if _ not in words:
            words.append(_)
    config.vocab_size=len(words)
    open_file(vocab_dir, mode='w').write('\n'.join(words) + '\n')


def read_vocab(vocab_dir):
    """读取词汇表"""
    # words = open_file(vocab_dir).read().strip().split('\n')
    with open_file(vocab_dir) as fp:
        # 如果是py2 则每个值都转化为unicode
        words = [native_content(_.strip()) for _ in fp.readlines()]
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id


def read_category():
    """读取分类目录，固定"""
    categories=config.categories

    cat_to_id=config.categories2id

    return categories, cat_to_id


def to_words(content, words):
    """将id表示的内容转换为文字"""
    return ''.join(words[x] for x in content)
    # return ' '.join(words[x] for x in content)


def process_file(filename, word_to_id, cat_to_id, max_length=600):
    """将文件文本内容转换为id表示"""
    data_text,data_subject,labels= read_file(filename)

    text_id,subject_id,label_id = [], [], []
    for i in range(len(data_text)):
        text_id.append([word_to_id[x] for x in data_text[i] if x in word_to_id])
        subject_id.append(word_to_id[data_subject[i]])
        label_id.append(int(labels[i]))

    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = kr.preprocessing.sequence.pad_sequences(text_id, max_length)
    y_pad = kr.utils.to_categorical(label_id, num_classes=len(cat_to_id))  # 将标签转换为one-hot表示
    subject_id=np.array(subject_id)
    return x_pad, y_pad, subject_id[:,np.newaxis]


def batch_iter(x, y,sub, batch_size=64):
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))  #打乱数据
    x_shuffle = x[indices]  #numpy索引用法
    y_shuffle = y[indices]
    sub_shuffle=sub[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id],sub_shuffle[start_id:end_id]
