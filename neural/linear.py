from __future__ import print_function

from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
# from keras.callbacks import EarlyStopping
# from keras.callbacks import Callback
from keras.models import Sequential
from keras.utils import plot_model
import matplotlib.pyplot as plt
from keras import backend as K
from datetime import datetime
from skimage import io
import numpy as np


BASE_IMG_PATH = 'data/modtrain-d148-crop-b4'
BATCH_SIZE = 128

# IMG_WIDTH = 150
# IMG_HEIGHT = 150
# IMG_WIDTH = 192
# IMG_HEIGHT = 192

IMG_HEIGHT = 148
IMG_WIDTH = 148


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
        imgs = [io.imread(x_paths[i % mod]) for i in range(idx, batch+idx)]
        trgs = [y_target[i % mod] for i in range(idx, batch+idx)]
        x = np.array(imgs, dtype=np.float32)/255
        y = np.array(trgs, dtype=np.float32)
        yield (x, y)
        idx = (idx+batch) % mod


def plot_history(history, path, file_time):

    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    mae_list = [s for s in history.history.keys() if 'mean' in s and 'val' not in s]
    val_mae_list = [s for s in history.history.keys() if 'mean' in s and 'val' in s]

    if len(loss_list) == 0:
        print('Loss is missing in history')
        return

    # As loss always exists
    epochs = range(1, len(history.history[loss_list[0]]) + 1)

    # Loss
    plt.figure(1)
    for l in loss_list:
        plt.plot(epochs, history.history[l], 'b',
                 label='Training loss (' + str(str(format(history.history[l][-1], '.5f')) + ')'))
    for l in val_loss_list:
        plt.plot(epochs, history.history[l], 'g',
                 label='Validation loss (' + str(str(format(history.history[l][-1], '.5f')) + ')'))

    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.savefig(path + '_loss_' + file_time + '.png')

    if len(mae_list) > 0:
        # Accuracy
        plt.figure(2)
        for l in mae_list:
            plt.plot(epochs, history.history[l], 'b',
                     label='Mean Absolute Error (' + str(format(history.history[l][-1], '.5f')) + ')')
        for l in val_mae_list:
            plt.plot(epochs, history.history[l], 'g',
                     label='Validation MAE (' + str(format(history.history[l][-1], '.5f')) + ')')

        plt.title('Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Mean Absolute Error')
        plt.legend()

        plt.savefig(path+'_mae_'+file_time+'.png')


def save_model(path, model, history):
    file_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    model.save_weights(path+'_weights_'+file_time+'.h5')
    model.save(path+'_model_'+file_time+'.h5')
    plot_model(model, to_file=path+'_summary'+file_time+'.png', show_shapes=True)
    plot_history(history, path, file_time)


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

    train_paths, train_ages, val_paths, val_ages = sep_paths()

    steps_per_epoch = (len(train_paths)+len(val_paths)) // BATCH_SIZE
    validation_steps = len(val_paths) // BATCH_SIZE
    epochs = 2

    model = create_model()
    history = model.fit_generator(generate_data(train_paths, train_ages, BATCH_SIZE),
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=epochs,
                                  verbose=1,
                                  validation_data=generate_data(val_paths, val_ages, BATCH_SIZE),
                                  validation_steps=validation_steps)

    save_model('data/ages', model, history)


if __name__ == '__main__':
    run_linear()

# end of file
