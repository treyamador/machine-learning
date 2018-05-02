from __future__ import print_function

from keras.layers import Dense, Dropout, Flatten, Activation, BatchNormalization, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from keras.optimizers import Adam, SGD
from keras.models import Sequential, Model
from keras.utils import plot_model
import matplotlib.pyplot as plt
from keras import backend as K
from datetime import datetime
from scipy import ndimage
import numpy as np
import os

from keras.applications.vgg16 import VGG16


BASE_IMG_PATH = 'data/modtrain-d224-crop-b3'
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
    mod, idx = len(x_paths), 0
    while True:
        imgs = [ndimage.imread(x_paths[i % mod]) for i in range(idx, batch+idx)]
        trgs = [y_target[i % mod] for i in range(idx, batch+idx)]
        x = np.array(imgs, dtype=np.float32) / PIXEL_NORMAL
        y = np.array(trgs, dtype=np.float32)
        yield (x, y)
        idx = (idx+batch) % mod


def plot_one(history, path, fig_num, metric):

    metric_list = [s for s in history.history.keys() if metric in s and 'val' not in s]
    val_list = [s for s in history.history.keys() if metric in s and 'val' in s]

    if len(metric_list) > 0:
        epochs = range(1, len(history.history[metric_list[0]]) + 1)
        plt.figure(fig_num)

        for l in metric_list:
            plt.plot(epochs, history.history[l], 'b',
                     label='training '+metric+' (' + str(str(format(history.history[l][-1], '.5f')) + ')'))
        for l in val_list:
            plt.plot(epochs, history.history[l], 'g',
                     label='validation '+metric+' (' + str(str(format(history.history[l][-1], '.5f')) + ')'))

        plt.title(metric)
        plt.xlabel('epochs')
        plt.ylabel(metric)
        plt.legend()

        plt.savefig(path+''+metric+'.png')


def plot_history(history, path):
    plot_one(history, path, 1, 'loss')
    plot_one(history, path, 2, 'mean_squared_error')
    plot_one(history, path, 3, 'acc')


def current_time():
    return datetime.now().strftime("%Y-%m-%d_%H:%M:%S")


def save_model(path, model, history):
    model.save_weights(path+'weights_final.h5')
    model.save(path+'model_final.h5')
    plot_history(history, path)
    plot_model(model, to_file=path+'model_summary.png', show_shapes=True)


def create_trivial():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(1))
    model.compile(loss='mse',
                  optimizer='sgd',
                  metrics=['mae', 'mse', 'acc'])
    return model


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
                  optimizer=Adam(),
                  metrics=['mae', 'mse', 'acc'])

    return model


def create_convolutional():

    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))

    model.compile(loss='mse',
                  optimizer=Adam(),
                  metrics=['mse', 'acc'])

    return model


def create_VGG16():

    base_model = VGG16(weights='imagenet',
                       include_top=False,
                       input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(1)(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=Adam(),
                  loss='mse',
                  metrics=['acc', 'mse'])

    return model


def train_VGG16(model, epochs):

    train_paths, train_ages, val_paths, val_ages = sep_paths()

    steps_per_epoch = len(train_paths) / BATCH_SIZE
    validation_steps = len(val_paths) / VALID_BATCH

    file_time = current_time()
    os.mkdir('models/'+file_time)

    callback_stopping = EarlyStopping(patience=3, monitor='val_loss')
    callback_checkpoint = ModelCheckpoint('models/model.vgg16.epoch.best.hdf5',
                                          save_best_only=True, monitor='val_loss')
    callback_csv = CSVLogger('models/'+file_time+'/run_log.csv', append=True)

    history = model.fit_generator(generate_data(train_paths, train_ages, BATCH_SIZE),
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=epochs,
                                  verbose=1,
                                  callbacks=[callback_checkpoint, callback_stopping, callback_csv],
                                  validation_data=generate_data(val_paths, val_ages, VALID_BATCH),
                                  validation_steps=validation_steps)

    save_model('models/' + file_time + '/', model, history)

    return model


def run_linear():

    model = create_VGG16()
    model = train_VGG16(model, 3)

    for layer in model.layers[:15]:
        layer.trainable = False
    for layer in model.layers[15:]:
        layer.trainable = True

    model.compile(loss='mse',
                  optimizer=SGD(lr=0.0001,
                                momentum=0.9),
                  metrics=['acc', 'mse'])

    train_VGG16(model, 100)


if __name__ == '__main__':
    run_linear()

# end of file
