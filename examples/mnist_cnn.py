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
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('mnist_cnn')

batch_size = 64
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

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
logger.info('x_train shape: {}'.format(x_train.shape))
logger.info('{} train samples'.format(x_train.shape[0]))
logger.info('{} test samples'.format(x_test.shape[0]))

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
logger.info('y_train = keras.utils.to_categorical(y_train, num_classes)')
y_test = keras.utils.to_categorical(y_test, num_classes)
logger.info('y_test = keras.utils.to_categorical(y_test, num_classes)')

model = Sequential()
logger.info('initiate sequential')
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
logger.info('add layer1')
model.add(Conv2D(64, (3, 3), activation='relu'))
logger.info('add layer2')
model.add(MaxPooling2D(pool_size=(2, 2)))
logger.info('add layer3')
model.add(Dropout(0.25))
logger.info('add layer4')
model.add(Flatten())
logger.info('add layer5')
model.add(Dense(128, activation='relu'))
logger.info('add layer6')
model.add(Dropout(0.5))
logger.info('add layer7')
model.add(Dense(num_classes, activation='softmax'))
logger.info('add layer8')

logger.info('good with create model')

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

logger.info('compile is fine')

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

logger.info('train should be okay')

score = model.evaluate(x_test, y_test, verbose=0)

logger.info('evaluation done')

logging.info('Test loss: {}'.format(score[0]))
print('Test accuracy:', score[1])
