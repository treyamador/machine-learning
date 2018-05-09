from __future__ import print_function

from keras.layers import Dense, Dropout, Flatten, Activation, BatchNormalization, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, TerminateOnNaN
from keras.optimizers import Adam, SGD
from keras.models import Sequential, Model, load_model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import backend as K
from datetime import datetime
from scipy import ndimage
import numpy as np
import os
import random

# from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50


# BASE_IMG_PATH = '../data/modtrain-d224'
BATCH_SIZE = 128
VALID_BATCH = BATCH_SIZE

PIXEL_NORMAL = 255.0

IMG_WIDTH = 224
IMG_HEIGHT = 224


if K.image_data_format() == 'channels_first':
    input_shape = (3, IMG_WIDTH, IMG_HEIGHT)
else:
    input_shape = (IMG_WIDTH, IMG_HEIGHT, 3)


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

    file_time = current_time()+'-gender-cnn'
    os.mkdir('models/'+file_time)

    train_datagen = ImageDataGenerator(
        rescale=1.0/255,
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    validation_datagen = ImageDataGenerator(rescale=1.0/255)

    train_generator = train_datagen.flow_from_directory(
        '../data/gender/train',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary')

    validation_generator = validation_datagen.flow_from_directory(
        '../data/gender/validation',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary')

    train_step_size = train_generator.n // train_generator.batch_size
    val_step_size = validation_generator.n // validation_generator.batch_size

    # callback_stopping = EarlyStopping(patience=3, monitor='val_loss')
    callback_checkpoint = ModelCheckpoint('models/'+file_time+'/model.resnet50.{epoch:02d}.loss-{val_loss:.2f}.hdf5',
                                          monitor='val_loss')
    callback_csv = CSVLogger('models/'+file_time+'/run_log.csv', append=True)
    callback_terminate_nan = TerminateOnNaN()

    model.compile(loss='binary_crossentropy',
                  optimizer=SGD(lr=0.0001, momentum=0.9),
                  metrics=['acc'])

    print('training step size', train_step_size)
    print('validation step size', val_step_size)

    model.fit_generator(train_generator,
                        epochs=epochs,
                        steps_per_epoch=train_step_size,
                        verbose=1,
                        callbacks=[callback_checkpoint,
                                   callback_csv,
                                   callback_terminate_nan],
                        validation_data=validation_generator,
                        validation_steps=val_step_size)

    return model


def run_linear():
    # model = create_ResNet50()
    model = load_model('models/2018-05-08_03:28:27-gender-generator/model.vgg16.38.loss-0.26.hdf5')
    for layer in model.layers[:-24]:
        layer.trainable = False
    for layer in model.layers[-24:]:
        layer.trainable = True
    train_ResNet50(model, 100)


if __name__ == '__main__':
    run_linear()

# end of file
