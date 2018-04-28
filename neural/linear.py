from __future__ import print_function

# from keras import utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
# from keras.callbacks import EarlyStopping
from keras import backend as K

from skimage import io
import numpy as np


BASE_IMG_PATH = 'data/modtrain-dSTD-crop-b4'
IMG_WIDTH = 150
IMG_HEIGHT = 150

if K.image_data_format() == 'channels_first':
    input_shape = (3, IMG_WIDTH, IMG_HEIGHT)
else:
    input_shape = (IMG_WIDTH, IMG_HEIGHT, 3)


def sep_paths():
    with open(BASE_IMG_PATH+'_target.csv') as f_obj:
        paths, ages = [], []
        for line in f_obj:
            path, age = line.split(',')
            age = int(age)
            paths.append(path)
            ages.append(age)
    paths = [BASE_IMG_PATH+'/'+x for x in paths]
    i = (4*len(paths))//5
    train_path, val_path = paths[:i], paths[i:]
    train_age, val_age = ages[:i], ages[:i]
    return train_path, train_age, val_path, val_age


def generate_file_imgs(x_paths, y_target):
    while True:
        for path, trgt in zip(x_paths, y_target):
            img = io.imread(path)
            img = np.array(img, dtype=np.float32)/255
            yield (img, trgt)


def create_model():

    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))

    model.compile(loss='mse',
                  optimizer='rmsprop',
                  metrics=['mae'])

    return model


def run_linear():

    batch_size = 128
    epochs = 2

    img_cols, img_rows = 384, 384
    channels = 3

    train_paths, train_ages, val_paths, val_ages = sep_paths()
    # generate_file_imgs(train_paths, train_ages)

    model = create_model()
    model.summary()


if __name__ == '__main__':
    run_linear()

