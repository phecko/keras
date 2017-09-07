# coding=utf8

'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# 转化为训练需要的结构,(个数,通过数,样本行,样本列)
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# create a model Sequential
model = Sequential()
# create a Convolution Layer,with 32 filters with 3x3 kernal, so output a (samples, 32, 28-2, 28-2)
# every filter create a dependence matrix layer
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))

# Con2D每一个kernal将使用由上一层得出的层数来计算,会把所有层叠在一个kernal里面,产生一个新层。这样理解为每一个kernal的每次一产生都是看过这个
# 位置的所有数据,不管前面用层把它分成多少层了
#
# 64 filters with 3x3 kernal, so output a (samples, 64, 26-2, 26-2)
# every filter will use a 25x3x3 matrix,it will with 225 input parameters
model.add(Conv2D(64, (3, 3), activation='relu'))

# create a MaxPooling Layer,every pool_size return (samples , 64 , 13, 13)
model.add(MaxPooling2D(pool_size=(2, 2)))
# 25% percent to drop
model.add(Dropout(0.25))
# flatten
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

# in the end use solfmax
model.add(Dense(num_classes, activation='softmax'))

# set the model compile
# optimizer：字符串（预定义优化器名）或优化器对象，参考优化器
# loss：字符串（预定义损失函数名）或目标函数，参考损失函数
# metrics：列表，包含评估模型在训练和测试时的网络性能的指标，典型用法是metrics=['accuracy']
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# start to train
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
