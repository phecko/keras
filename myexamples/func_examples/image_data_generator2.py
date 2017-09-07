# coding=utf8

from keras.preprocessing.image import ImageDataGenerator, load_img, array_to_img, img_to_array

import os

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2, #剪切变换程度
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
)

p = os.path.abspath("../../datas/")

img = load_img(p + '/dogsVscats/train/cats/cat.5.jpg') # this is a PIL image
x = img_to_array(img)  # a Numpy array (3, 150, 150)
x = x.reshape((1,) + x.shape) # to (1, 3, 150, 150)


# gen image
i = 0
for batch in datagen.flow(x,
                          batch_size = 1,
                          save_to_dir=p + '/dogsVscats/preview',
                          save_prefix='cat',
                          save_format='jpeg',
                          ):
    i += 1
    if i > 20:
        break




