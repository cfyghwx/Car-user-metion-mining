# -*- coding: utf-8 -*-

import tensorflow as tf

'''
tf.dynamic_rnn参数:
https://blog.csdn.net/u010223750/article/details/71079036

tf.nn.dynamic_rnn的输出outputs和state含义:
https://blog.csdn.net/u010960155/article/details/81707498

双向rnn (BiRNN):
https://blog.csdn.net/u012436149/article/details/71080601
https://blog.csdn.net/wuzqChom/article/details/75453327
https://blog.csdn.net/wuzqChom/article/details/75453327
拼接https://blog.csdn.net/lijin6249/article/details/78955175
'''


class TextRNN(object):
    """文本分类，RNN模型"""
    def __init__(self, config):
        self.config = config

        # 三个待输入的数据
        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.input_sub=tf.placeholder(tf.int32, [None, 1], name='input_sub')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.rnn()

    def rnn(self):
        """rnn模型"""

        def lstm_cell():   # lstm核
            return tf.contrib.rnn.BasicLSTMCell(self.config.hidden_dim, state_is_tuple=True)

        def gru_cell():  # gru核
            return tf.contrib.rnn.GRUCell(self.config.hidden_dim)

        def dropout(): # 为每一个rnn核后面加一个dropout层
            if (self.config.rnn == 'lstm'):
                cell = lstm_cell()
            else:
                cell = gru_cell()
            return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

        # 词向量映射（创建词向量矩阵，根据索引获取词向量）
        with tf.device('/cpu:0'):
            # init = tf.constant_initializer(init_embedding)
            # tf.get_variable(name, shape=None, dtype=None, initializer=None, regularizer=None, trainable=True, collections=None, caching_device=None, partitioner=None, validate_shape=True, custom_getter=None)
            embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim],trainable=True)
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)
            embedding_sub = tf.nn.embedding_lookup(embedding, self.input_sub)
            embedding_sub=tf.reshape(embedding_sub,[-1,self.config.embedding_dim])
        '''
        with tf.name_scope("rnn"):
            # 多层rnn网络
            cells = [dropout() for _ in range(self.config.num_layers)]
            rnn_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

            _outputs, _ = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=embedding_inputs, dtype=tf.float32)
            # _outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell=rnn_cell, inputs=embedding_inputs, dtype=tf.float32)
            last = _outputs[:, -1, :]  # 取最后一个时序输出作为结果
        '''
        with tf.name_scope("rnn"):
            # num_layers层双向rnn网络
            cells_forward = [dropout() for _ in range(self.config.num_layers)]
            cells_backward = [dropout() for _ in range(self.config.num_layers)]
            cells_forward = tf.contrib.rnn.MultiRNNCell(cells_forward, state_is_tuple=True)
            cells_backward = tf.contrib.rnn.MultiRNNCell(cells_backward, state_is_tuple=True)

            _outputs, _ = tf.nn.bidirectional_dynamic_rnn(cells_forward,cells_backward, inputs=embedding_inputs, dtype=tf.float32)
            # last = _outputs[:, -1, :]  # 取最后一个时序输出作为结果
            # out  = np.concatenate((_[0][-1],_[1][-1]),axis = 1)
            # last=tf.concat([_[0][-1],_[1][-1]],1)
            out=tf.add(_[0][-1],_[1][-1])
            last=tf.concat([out,embedding_sub],1)

        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            fc = tf.layers.dense(last, self.config.hidden_dim, name='fc1')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)

            # 分类器
            self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2')
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别

        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
