from __future__ import print_function

from keras.layers import Dense, Dropout, Flatten, Activation, BatchNormalization, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, TerminateOnNaN, TensorBoard
from keras.optimizers import Adam, SGD
from keras.models import Sequential, Model, load_model
from keras import backend as K
from datetime import datetime
from scipy import ndimage
import numpy as np
import os
import random

from keras.applications.vgg16 import VGG16


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
    train_age, val_age = ages[:i], ages[i:]
    return train_path, train_age, val_path, val_age


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


def create_VGG16():

    base_model = VGG16(weights='imagenet',
                       include_top=False,
                       input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1)(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model


def train_VGG16(model, epochs):

    train_paths, train_ages, val_paths, val_ages = sep_paths()
    steps_per_epoch = len(train_paths) / BATCH_SIZE
    validation_steps = len(val_paths) / VALID_BATCH

    attrib = 'densenet'
    model_type = 'vgg16'
    file_time = current_time()+'-'+attrib
    os.mkdir('models/'+file_time)

    with open('models/'+file_time+'/attrib.txt', 'wt') as file_writer:
        file_writer.write('used '+model_type+'.'+attrib+'.py\n')

    # callback_stopping = EarlyStopping(patience=10, monitor='val_loss')
    callback_checkpoint = ModelCheckpoint('models/'+file_time+'/model.' + model_type + '-' + attrib +
                                          '.{epoch:02d}.mse-{val_loss:.2f}.hdf5',
                                          monitor='val_loss')
    callback_csv = CSVLogger('models/'+file_time+'/run_log.csv', append=True)
    callback_terminate_nan = TerminateOnNaN()
    callback_tensor_board = TensorBoard(log_dir='./Graph', histogram_freq=0,
                                        write_graph=True, write_images=True)

    model.compile(loss='mse',
                  optimizer=SGD(lr=0.0001, momentum=0.9),
                  metrics=['acc', 'mse'])

    history = model.fit_generator(generate_data(train_paths, train_ages, BATCH_SIZE),
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=epochs,
                                  verbose=1,
                                  callbacks=[callback_checkpoint,
                                             # callback_stopping,
                                             callback_csv,
                                             callback_terminate_nan,
                                             callback_tensor_board],
                                  validation_data=generate_data(val_paths, val_ages, VALID_BATCH),
                                  validation_steps=validation_steps)

    with open('models/'+file_time+'/job_completed.txt', 'wt') as file_writer:
        file_writer.write('job finished gracefully\n')

    return model


def run_linear():
    model = create_VGG16()
    for layer in model.layers[:-6]:
        layer.trainable = False
    for layer in model.layers[-6:]:
        layer.trainable = True
    train_VGG16(model, 100)


if __name__ == '__main__':
    run_linear()

# end of file
