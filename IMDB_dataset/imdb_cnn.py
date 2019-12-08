# -*- coding: utf-8 -*-
'''
# Trains a Bidirectional LSTM on the IMDB sentiment classification task.
90s / epoch I5 CPU
验证集准确性： val_acc:  0.8919
'''

# This references from https://github.com/keras-team/keras/blob/master/examples/imdb_cnn.py

from __future__ import print_function, division
import tensorflow as tf
from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Embedding, Activation
from tensorflow.python.keras.layers import Conv1D, GlobalMaxPooling1D
from tensorflow.python.keras.datasets import imdb
import numpy as np

# 为了方便，我们直接假设有 20000 个单词
word_nums = 20000
maxlen = 400
batch_size = 32
embedding_dims = 128
filter_nums = 128
kernel_size = 3
hidden_dims = 64
epochs = 2

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=word_nums)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')


print('下面我们对数据集进行截断、补齐操作，保证 max_len = 400')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')

model = Sequential()
model.add(Embedding(word_nums, embedding_dims, input_length=maxlen))
model.add(Dropout(0.2))
# 开始利用 Convolution1D 来添加 filters
model.add(Conv1D(filters=filter_nums,
                 kernel_size=kernel_size,
                 padding="valid",
                 activation="relu",
                 strides=1))

# 随后利用 MaxPooling 来对所有的 filter 进行过滤
model.add(GlobalMaxPooling1D())

# 再之后接一个线性层
model.add(Dense(hidden_dims))
model.add(Dropout(0.2))
model.add(Activation('relu'))

# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print(model.summary())

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))












