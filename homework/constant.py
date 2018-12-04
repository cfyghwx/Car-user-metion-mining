#coding=utf-8

class Config(object):
    subject_labels=['动力','价格','内饰','配置','安全性','外观','操控','油耗','空间','舒适性']
    subject2id=dict(zip(subject_labels, range(len(subject_labels))))
    id2subject=dict(zip(range(len(subject_labels)),subject_labels))
    categories = ['中立','正向','负向']
    categories2id={'中立':0,'正向':1,'负向':-1}

    vocab_threshold=5
    padding_length=100

    """RNN配置参数"""
    # 模型参数
    embedding_dim = 128      # 词向量维度
    seq_length = 600        # 序列长度
    num_classes = 3        # 类别数
    vocab_size = 20000       # 词汇表达小

    num_layers= 1           # 隐藏层层数
    hidden_dim = 128        # 隐藏层神经元
    rnn = 'gru'             # lstm 或 gru

    dropout_keep_prob = 0.8 # dropout保留比例
    learning_rate = 1e-3    # 学习率

    batch_size = 128         # 每批训练大小
    num_epochs = 10          # 总迭代轮次

    print_per_batch = 100    # 每多少轮输出一次结果
    save_per_batch = 10      # 每多少轮存入tensorboard
    

config=Config()