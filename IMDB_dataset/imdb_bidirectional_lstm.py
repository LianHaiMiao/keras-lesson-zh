# -*- coding: utf-8 -*-
'''
# Trains a Bidirectional LSTM on the IMDB sentiment classification task.
123s / epoch I5 CPU
验证集准确性： val_acc:  0.7877
'''

# This references from https://github.com/keras-team/keras/blob/master/examples/imdb_bidirectional_lstm.py

from __future__ import print_function, division
import tensorflow as tf
from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from tensorflow.python.keras.datasets import imdb
import numpy as np

# 为了方便，我们直接假设有 20000 个单词
word_nums = 20000
# 数据集最大长度
maxlen = 100
batch_size = 32
embedding_dims = 128

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=word_nums)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('下面我们对数据集进行截断、补齐操作，保证 max_len = 100')
# "post" 表示从最后进行截断、或者补齐
x_train = sequence.pad_sequences(sequences=x_train,
                                 maxlen=maxlen,
                                 padding="post",
                                 truncating="post")

x_test = sequence.pad_sequences(sequences=x_test,
                                 maxlen=maxlen,
                                 padding="post",
                                 truncating="post")

print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

y_train = np.array(y_train)
y_test = np.array(y_test)

# 构建模型
model = Sequential()
model.add(Embedding(word_nums, embedding_dims, input_length=maxlen))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid"))

# 编译模型图
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

print('Train...')

# 开始训练
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=4,
          validation_data=[x_test, y_test])






