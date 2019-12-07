from __future__ import print_function
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
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


# 通过 dataset API 构建数据流
# <DatasetV1Adapter shapes: ((28, 28, 1), (10,)), types: (tf.float32, tf.float32)>
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(500).batch(32)

test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = test_dataset.shuffle(500).batch(32)

# 这里创建的模型，我们直接使用一个 function 来进行创建而不是使用 Sequential()

def cnn_model(input_shape, num_classes=10):
    """
    # Arguments
        input_shape (tensor): shape of input image tensor
        num_classes (int): number of classes (CIFAR10 has 10)
    # Returns
        model (Model): Keras model instance
    """
    # 初始化一个 Keras 输入
    inputs = Input(shape=input_shape)
    # conv-layer-1
    x = Conv2D(32, kernel_size=(3, 3), activation='relu')(inputs)
    # conv-layer-2
    x = Conv2D(64, kernel_size=(3, 3), activation="relu")(x)
    # pool-layer-1
    x = MaxPool2D(pool_size=(2, 2))(x)
    # dropout layer
    x = Dropout(0.25)(x)
    # Flatten Layer
    x = Flatten()(x)
    # dense-layer-1
    x = Dense(128, activation='relu')(x)
    # dropout layer
    y = Dropout(0.5)(x)
    # dense-layer-2
    outputs = Dense(num_classes,
                    activation="softmax")(y)
    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model

model = cnn_model(input_shape=(img_rows, img_cols, 1), num_classes=10)


# 构建模型图，使用 model.compile
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

print(model.summary())

# 开始训练
model.fit(train_dataset,
          epochs=epochs,
          verbose=2,
          validation_data=(x_test, y_test))


score = model.evaluate(test_dataset, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])





