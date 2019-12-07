from __future__ import print_function
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras import backend as K

'''Trains a simple convnet on the MNIST dataset with dataset, estimator and keras API
参照了这篇文章： https://zhuanlan.zhihu.com/p/66872472
'''
# global parameters
# input image dimensions
img_rows, img_cols = 28, 28
BATCH_SIZE = 128
EPOCH = 2


class testclsHook(tf.train.SessionRunHook):

    def begin(self):
        """再创建会话之前调用
        调用begin()时，default graph会被创建，
        可在此处向default graph增加新op,begin()调用后，default graph不能再被修改
        """
        print("first")
        pass

    def after_create_session(self, session, coord):  # pylint: disable=unused-argument
        print("2end")

        pass

    def before_run(self, run_context):  # pylint: disable=unused-argument
        print("third")

        return None

    def after_run(self,
                  run_context,  # pylint: disable=unused-argument
                  run_values):  # pylint: disable=unused-argument
        """调用在每个sess.run()之后
        参数run_values是befor_run()中要求的op/tensor的返回值；
        可以调用run_context.qeruest_stop()用于停止迭代
        sess.run抛出任何异常after_run不会被调用
        Args:
          run_context: A `SessionRunContext` object.
          run_values: A SessionRunValues object.
        """
        print("4th")
        pass

    def end(self, session):  # pylint: disable=unused-argument
        """在会话结束时调用
        end()常被用于Hook想要执行最后的操作，如保存最后一个checkpoint
        如果sess.run()抛出除了代表迭代结束的OutOfRange/StopIteration异常外，
        end()不会被调用
        Args:
          session: A TensorFlow Session that will be soon closed.
        """
        print("5th")
        pass


# 通过 dataset API 构建数据流
def input_fn_builder(x, y, batch_size=32, shuffle=500, epochs=12, is_train=True):
    '''
       创建 输入函数闭包
       返回的可以是一个 Dataset 处理后的数据，也可以是 dict({"feature": xxxx, "label": xxxx})
    '''
    def input_fn():
        dataset = tf.data.Dataset.from_tensor_slices(({"images": x}, y))
        if is_train:
            dataset = dataset.shuffle(shuffle).repeat(epochs)
        dataset = dataset.batch(batch_size)
        return dataset
    return input_fn

def create_model(features, feature_columns, num_classes):
    # 只负责网络架构的创建
    #  ```python
    # price = numeric_column('price')
    # keywords_embedded = embedding_column(
    #     categorical_column_with_hash_bucket("keywords", 10K), dimensions=16)
    # columns = [price, keywords_embedded, ...]
    # features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
    # dense_tensor = input_layer(features, columns)
    # for units in [128, 64, 32]:
    #   dense_tensor = tf.compat.v1.layers.dense(dense_tensor, units, tf.nn.relu)
    # prediction = tf.compat.v1.layers.dense(dense_tensor, 1)
    # ```
    # input_layer = tf.feature_column.input_layer(features, feature_columns)

    # features 就是存粹的数据，保存格式为 {'feature_name_1': value_1, 'feature_name_2': value_2, ...}
    # feature_columns 定义了我们如何对 feature_name_1, feature_name_2, ... 等等 feature_name 进行相应的操作。
    # 这一层的默认操作是 array_ops.concat(output_tensors, output_rank - 1)
    # 这样的操作会使得所有的 feature 经过各自的 column 处理之后 concat 在一起。
    input_layer = tf.feature_column.input_layer(features, feature_columns)
    # 接下来我们构建的网络的输入【这一部分我们最好使用 Keras 去搭建】
    inputs = tf.reshape(input_layer, [-1, img_cols, img_rows, 1])
    l2 = tf.keras.regularizers.l2(l=0.01)
    # 初始化一个 Keras 输入
    # conv-layer-1
    x = Conv2D(32, kernel_size=(3, 3), activation='relu', name='conv1')(inputs)
    # conv-layer-2
    x = Conv2D(64, kernel_size=(3, 3), activation="relu", name='conv2')(x)
    # pool-layer-1
    x = MaxPool2D(pool_size=(2, 2))(x)
    # dropout layer
    x = Dropout(0.25)(x)
    # Flatten Layer
    flat = Flatten()(x)
    # dense-layer-1
    dens1 = Dense(128, activation='relu', name='dense1', kernel_regularizer=l2)(flat)
    # 由于我们使用这里的 label 是 batch*1 格式的，所以我们使用 softmax_cross_entropy_with_logits 作为 loss
    # 使用该 loss 不需要在最后一层加 softmax
    logits = Dense(num_classes, name='dense_output')(dens1)
    return logits

def  model_fn_builder(lr):
    # 我们使用这个方法来构建 estimator的model_fn
    def model_fn(features, labels, mode=tf.estimator.ModeKeys.TRAIN, params=None, config=None):
        '''
            features: from input_fn的返回  切记返回的顺序
            labels： from input_fn 的返回  切记返回的顺序
            mode: tf.estimator.ModeKeys实例的一种
            params: 在初始化estimator时 传入的参数列表，dict形式,或者直接使用self.params也可以
            config:初始化estimator时 的 Runconfig
        '''
        logits = create_model(features, params['feature_columns'], params['output_cls'])
        # 预测的类别
        predict_cls = tf.argmax(input=logits, axis=1)
        # 预测的类别对应的概率
        predict_pro = tf.nn.softmax(logits=logits, axis=1)

        # TRAIN 和 EVAL 需要 (features, label)
        # PREDICT 走另外一条链路，不需要 label，只需要 features
        is_predict = (mode == tf.estimator.ModeKeys.PREDICT)

        if not is_predict:
            # TRAIN 和 EVAL
            loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
            tf.summary.scalar('loss1', tf.squeeze(loss))

            def metric_fn(labels, predictions):
                """ 定义评价指标
                :param labels: 真实值
                :param predictions: 预测值
                :return:
                """
                accuracy, accuracy_update = tf.metrics.accuracy(labels=labels, predictions=predictions,
                                                                name='image_accuracy')
                recall, recall_update = tf.metrics.recall(labels=labels, predictions=predictions, name='image_recall')
                precision, precision_update = tf.metrics.precision(labels=labels, predictions=predictions,
                                                                   name='image_precision')
                return {
                    'accuracy': (accuracy, accuracy_update),
                    'recall': (recall, recall_update),
                    'precision': (precision, precision_update)
                }

            # 构建评价指标
            metrics = metric_fn(labels, predict_cls)

            if mode == tf.estimator.ModeKeys.EVAL:
                return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=metrics)

            # 训练 op
            train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss=loss,
                                                                         global_step=tf.train.get_global_step())

            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, eval_metric_ops=metrics)

        else:
            predictions = {'predict_cls': predict_cls, 'predict_pro': predict_pro}
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    return model_fn

if __name__ == '__main__':
    num_classes = 10

    # x_train (60000, 28, 28); y_train (60000,)
    # x_test (10000, 28, 28); y_test (10000,)
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # 通常来说 Keras 的后台如果是 tensorflow 一般都是 channels_last
    # 将数据转换成 batch*img_rows*img_cols*channel 的格式
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

    y_train = y_train.astype('int32')
    y_test = y_test.astype('int32')

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # 准备数据环节结束
    print("label data type", type(y_train[0]))


    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # 一些参数
    model_dir = r'./model/'
    params = {}
    feature_columns = [tf.feature_column.numeric_column('images', shape=(img_cols, img_rows))]

    params['feature_columns'] = feature_columns
    params['output_cls'] = num_classes

    print(feature_columns)

    # 开始构建
    config = tf.estimator.RunConfig(save_checkpoints_steps=100)

    estimator = tf.estimator.Estimator(model_fn=model_fn_builder(0.001), model_dir=model_dir, params=params,
                                       config=config)

    # tensors_to_log = {"test_hook": "predict_pro"}

    # logging_hook = tf.train.LoggingTensorHook(
    #     tensors=tensors_to_log, every_n_iter=50)

    train = estimator.train(
        input_fn=input_fn_builder(x=x_train, y=y_train, batch_size=BATCH_SIZE, epochs=EPOCH, is_train=True),
        steps=10000)
    print(train)

    test = estimator.evaluate(
        input_fn=input_fn_builder(x=x_test, y=y_test, batch_size=BATCH_SIZE, epochs=EPOCH, is_train=False))

    print(test)