# coding=utf8

from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import keras.backend  as K
from keras.preprocessing.image import ImageDataGenerator

model = Sequential()

input_shape = (3, 150, 150)
if K.image_data_format() == "channels_first":
    input_shape = (3, 150, 150)
else:
    input_shape = (150, 150, 3)

# add 3 Convolution Layer

model.add(Convolution2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


# add full net

model.add(Flatten())
model.add(Dense(64))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)


test_datagen = ImageDataGenerator(
    rescale=1./255,
)


train_generator = train_datagen.flow_from_directory(
    "../datas/dogsVscats/train",
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
)

validation_generator = test_datagen.flow_from_directory(
    "../datas/dogsVscats/validation",
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
)


model.fit_generator(
    train_generator,
    steps_per_epoch=2000,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=800,
)

model.save_weights('first_try.h5')
