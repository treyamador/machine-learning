from __future__ import print_function

from keras.layers import Dense, Dropout, Flatten, Activation, BatchNormalization, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, TerminateOnNaN
from keras.optimizers import Adam, SGD
from keras.models import Sequential, Model, load_model
from keras import backend as K
from datetime import datetime
from scipy import ndimage
import numpy as np
import os
import random

from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50


BASE_IMG_PATH = '../data/modtrain-d224'
BATCH_SIZE = 128
VALID_BATCH = BATCH_SIZE

PIXEL_NORMAL = 255.0

IMG_WIDTH = 224
IMG_HEIGHT = 224


if K.image_data_format() == 'channels_first':
    input_shape = (3, IMG_WIDTH, IMG_HEIGHT)
else:
    input_shape = (IMG_WIDTH, IMG_HEIGHT, 3)


def sep_paths():
    with open(BASE_IMG_PATH+'_gender_target.csv') as f_obj:
        paths, genders = [], []
        next(f_obj)
        for line in f_obj:
            path, gender = line.split(',')
            gender = int(gender)
            paths.append(path)
            genders.append(gender)
    paths = [BASE_IMG_PATH+'/'+x for x in paths]
    i = (4*len(paths))//5
    train_path, val_path = paths[:i], paths[i:]
    train_gen, val_gen = genders[:i], genders[i:]
    return train_path, train_gen, val_path, val_gen


def generate_data(x_paths, y_target, batch):
    mod = len(x_paths)
    while True:
        s = random.sample(range(0, mod), batch)
        imgs = [ndimage.imread(x_paths[i]) for i in s]
        trgs = [y_target[i] for i in s]
        x = np.array(imgs, dtype=np.float32) / PIXEL_NORMAL
        y = np.array(trgs, dtype=np.float32)
        yield (x, y)


def current_time():
    return datetime.now().strftime("%Y-%m-%d_%H:%M:%S")


def create_ResNet50():

    base_model = ResNet50(weights='imagenet',
                          include_top=False,
                          input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model


def train_ResNet50(model, epochs):

    train_paths, train_ages, val_paths, val_ages = sep_paths()
    steps_per_epoch = len(train_paths) / BATCH_SIZE
    validation_steps = len(val_paths) / VALID_BATCH

    file_time = current_time()+'-gender-frozen'
    os.mkdir('models/'+file_time)

    # callback_stopping = EarlyStopping(patience=3, monitor='val_loss')
    callback_checkpoint = ModelCheckpoint('models/'+file_time+'/model.resnet50.{epoch:02d}.loss-{val_loss:.2f}.hdf5',
                                          monitor='val_loss')
    callback_csv = CSVLogger('models/'+file_time+'/run_log.csv', append=True)
    callback_terminate_nan = TerminateOnNaN()

    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(),
                  metrics=['acc'])

    history = model.fit_generator(generate_data(train_paths, train_ages, BATCH_SIZE),
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=epochs,
                                  verbose=1,
                                  callbacks=[callback_checkpoint,
                                             # callback_stopping,
                                             callback_csv,
                                             callback_terminate_nan],
                                  validation_data=generate_data(val_paths, val_ages, VALID_BATCH),
                                  validation_steps=validation_steps)

    return model


def run_linear():
    model = create_ResNet50()
    for layer in model.layers[:-2]:
        layer.trainable = False
    for layer in model.layers[-2:]:
        layer.trainable = True
    train_ResNet50(model, 100)


if __name__ == '__main__':
    run_linear()

# end of file
