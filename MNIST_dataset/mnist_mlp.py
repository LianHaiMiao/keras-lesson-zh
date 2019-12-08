import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop

# This references from https://github.com/keras-team/keras/blob/master/examples/mnist_mlp.py
# 这个主要是记录一下重新学 Keras 的点点滴滴。

# print(tf.__version__)
# 1.14.0

batch_size = 128
num_classes = 10
epochs = 1

# x_train (60000, 28, 28); y_train (60000,)
# x_test (10000, 28, 28); y_test (10000,)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# 归一化
x_train /= 255
x_test /= 255

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# 由于我们这里进行的是分类任务，我们需要将 label 转换成 one-hot encoding 的形式。
# (60000, 10)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# 开始构建模型
model = Sequential()
model.add(Dense(512, activation="relu", input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

# 打印出模型概述信息
model.summary()

# 模型编译
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

# 模型训练
# x: Input Features
# y: Label
# verbose: 训练进度显示方式
# 0 = silent, 1 = progress bar, 2 = one line per epoch.
history = model.fit(x=x_train, y=y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))


score = model.evaluate(x_test, y_test, verbose=0)
print(score)
print('Test loss:', score[0])
print('Test accuracy:', score[1])