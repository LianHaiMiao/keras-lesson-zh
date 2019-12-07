from __future__ import print_function
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras import backend as K

'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# x_train (60000, 28, 28); y_train (60000,)
# x_test (10000, 28, 28); y_test (10000,)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 通常来说 Keras 的后台如果是 tensorflow 一般都是 channels_last
# 将数据转换成 batch*img_rows*img_cols*channel 的格式
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# 准备数据环节结束

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# 多分类问题，我们使用 cross_entropy
# 首先将 label 转换成 one hot encoding
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# 通过 dataset API 构建数据流。由于我们后续会在 train 的时候会使用 epoch 这个参数。
# 所以构建数据流的时候，不需要使用 .repeat(epoch) 这个值
# <DatasetV1Adapter shapes: ((28, 28, 1), (10,)), types: (tf.float32, tf.float32)>
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(500).batch(32)

test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = test_dataset.shuffle(500).batch(32)

# 通过 dataset API 构建数据流

# 构建模型，暂时直接使用 Sequential() 来构建
model = Sequential()
# CNN 计算：output_size = 1 + (input_size + 2*padding - kernel_size) / stride
# first-layer 需要使用 input_shape 这个超参
# (batch, 28, 28, 1) -> (batch, 26, 26, 32)
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
# (batch, 24, 24, 32) -> (batch, 26, 26, 64)
model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
# (batch, 26, 26, 64) -> (batch, 13, 13, 64)
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
# (batch, 13, 13, 64) -> (batch, 10816)
model.add(Flatten())
# (batch, 10816) -> (batch, 128)
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
# (batch, 128) -> (batch, 10)
model.add(Dense(num_classes, activation="softmax"))

# 构建模型图，使用 model.compile
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])


# 开始训练
model.fit(train_dataset,
          epochs=epochs,
          verbose=2,
          validation_data=(x_test, y_test))


score = model.evaluate(test_dataset, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])





