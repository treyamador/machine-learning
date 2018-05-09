# import print function to ensure new print functions run with previous versions of python
from __future__ import print_function
# import the various layers from keras
# only a handful are used in this particular code
from keras.layers import Dense, Dropout, Flatten, Activation, BatchNormalization, GlobalAveragePooling2D
# import the convolutional layers, which are not used here
from keras.layers import Conv2D, MaxPooling2D
# import callback functions which are executed upon the end of each epoch
# EarlyStopping prevents neural net from continuing too long
# ModelCheckpoint allows saving of models or weights
# CSVLogger allows saving of specific datapoints into csv file
# TerminateOnNaN allows stopping when NaN (including infinity) is encountered
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, TerminateOnNaN
# import major optimizers, Adam and Stochastic Gradient Descent
from keras.optimizers import Adam, SGD
# import model-building classes
# Model can be used to load and modify model with Keras's functional API
from keras.models import Sequential, Model, load_model
# import image generation, which prevents the model from overfitting with a limited set of image data
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
# backend can be used to check the order of channels
from keras import backend as K
# used to keep track of time information
from datetime import datetime
# import image modification/loading library
from scipy import ndimage
# import numpy for fast-processing of arrays
import numpy as np
# import operating system to allow file manipulation
import os
# import random to allow 'shuffling' of image data
import random
# import the VGG16 model as included in Keras
from keras.applications.vgg16 import VGG16

# define batch size
BATCH_SIZE = 128
# define batch size for validation data set
VALID_BATCH = BATCH_SIZE
# define the Normal for color information
# i.e., each pixel will be divided by its maximum value
# to keep the values low and prevent information loss
PIXEL_NORMAL = 255.0
# define default width of image
IMG_WIDTH = 224
# define default height of image
IMG_HEIGHT = 224

# check if image is 'channels_first'
# meaning that the color channel is the first element of the shape attribute
if K.image_data_format() == 'channels_first':
    # if so, define the input shape as channels first
    input_shape = (3, IMG_WIDTH, IMG_HEIGHT)
# otherwise
else:
    # define the image shape as channels last,
    # which is the default mode for TensorFlow
    input_shape = (IMG_WIDTH, IMG_HEIGHT, 3)

# define a function to retrieve a formatted version of the current time to seconds
def current_time():
    # return formatted string of current time, to the seconds
    return datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

# function to initialize a new VGG16 architecture
def create_VGG16():
    # define the base model, pretrained on imagenet, and without its basic top layer
    # leverages transfer learning to enhance image recognition
    base_model = VGG16(weights='imagenet',
                       include_top=False,
                       input_shape=input_shape)
    # assign the output of the model to a holder variable
    # to be passed to the other layers of the API
    x = base_model.output
    # create global average pooling layer, to allow flattening of image data
    x = GlobalAveragePooling2D()(x)
    # create dense layer with relu activation
    # 1024 was chosen in an attempt to allow sufficient
    # information to be contained
    # without allowing the holding of noise or extraneous information
    x = Dense(1024, activation='relu')(x)
    # dropout inlcuded to prevent association of unrelated information
    # prevents 'conspiracies' between neurons
    x = Dropout(0.3)(x)
    # another dense layer is added
    x = Dense(1024, activation='relu')(x)
    # another dropout layer is added
    x = Dropout(0.3)(x)
    # the dense output layer is given sigmoid activation
    # which will tell the probability that a subject is male
    # i.e., a 1 denotes maleness, a 0 denotes femaleness
    predictions = Dense(1, activation='sigmoid')(x)
    # place the inputs and outputs into the model class
    model = Model(inputs=base_model.input, outputs=predictions)
    # return the modified model
    return model


def train_VGG16(model, epochs):
    # a function to allow the training of a given neural network
    # create a file name with the current time to prevent
    #   confusion with other experiments of this architecture
    file_time = current_time()+'-gender-cnn'
    # create directory with the given name 
    #   including date and name of experiment
    os.mkdir('models/'+file_time)
    # create an image generator for the training data
    # the image is rescaled so pixel values are between 0 and 1
    # shearing distorts the image
    # zooming closes-in on some aspect of the image
    # rotation range rotates the image
    # width and height shift move the image left or right,
    #   or up or down
    # horizontal flip flips the image
    # the purpose of image generators is to stifle overfitting
    #   by preventing the neural net from recognizing attributes
    #   specific to the images, and not the conditions to which
    #   the images belong
    train_datagen = ImageDataGenerator(
        rescale=1.0/255,
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
    # a validation generator is created which does not distort the images
    #   but does resize the magnitude of the pixel data
    validation_datagen = ImageDataGenerator(rescale=1.0/255)
    # the train generator is created, taking images from the 
    #   male and female subfolders of the defined directories
    #   the image and batch size are defined,
    #   and binary mode is used given that there are
    #   two conditions, male or female
    train_generator = train_datagen.flow_from_directory(
        '../data/gender/train',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary')
    # a validation generator is created, much like the training generator
    validation_generator = validation_datagen.flow_from_directory(
        '../data/gender/validation',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary')
    # the step size is the number of batches in the testing data set
    train_step_size = train_generator.n // train_generator.batch_size
    # the step size is teh number of batches in the validation data set
    val_step_size = validation_generator.n // validation_generator.batch_size

    # early stopping allows network to terminate early if no progress is made
    # callback_stopping = EarlyStopping(patience=10, monitor='val_loss')
    # model checkpoint saves the model, which gives users the ability to later load the model into memory
    callback_checkpoint = ModelCheckpoint('models/'+file_time+'/model.vgg16.{epoch:02d}.loss-{val_loss:.2f}.hdf5',
                                          monitor='val_loss')
    # the csv logger accounts for loss data and accuracy
    callback_csv = CSVLogger('models/'+file_time+'/run_log.csv', append=True)
    # create callback to terminate when a loss value is encountered
    callback_terminate_nan = TerminateOnNaN()
    # compile the model
    # during the preliminary training, only the new layers are trainable
    #   the optimizer used was typically Adam
    #   However, during convolutional layer training,
    #   a lower learning rate is required such that 
    #   the weights of the convolutional layer are not skewed
    #   binary_crossentropy is log loss for binary category
    model.compile(loss='binary_crossentropy',
                  optimizer=SGD(lr=0.0001, momentum=0.9),
                  metrics=['acc'])

    # print the number of training steps to console
    print('training step size', train_step_size)
    # print the number of validation steps to console
    print('validation step size', val_step_size)

    # run the training of the model
    #   a train generator feeds images and target data to model
    #   epochs is the number of epochs used
    #   step sizes are the number of images to step
    #   the callback array is passed in
    #   the validation generator feeds validation images to train model
    model.fit_generator(train_generator,
                        epochs=epochs,
                        steps_per_epoch=train_step_size,
                        verbose=1,
                        callbacks=[callback_checkpoint,
                                   callback_csv,
                                   callback_terminate_nan],
                        validation_data=validation_generator,
                        validation_steps=val_step_size)

    # return the generated model
    return model


def run_linear():
    # create a model from scratch with imagenet loaded
    # model = create_VGG16()
    # alternatively, load a model trained by the user instead
    model = load_model('models/2018-05-08_16:47:49-gender-cnn/model.vgg16.37.loss-0.21.hdf5')
    # iterate through first 15 layers
    #   in the case of the first round of training,
    #   layers above max pooling are trainable
    #   After the first round, the convolutional layers are trained
    for layer in model.layers[:15]:
        # freeze layers to prevent training
        #   so the weights are not destroyed
        layer.trainable = False
    # iterate through layers above first 15
    for layer in model.layers[15:]:
        # set layers to trainable
        layer.trainable = True
    # run the training of the model for 100 epochs or more    
    train_VGG16(model, 100)


if __name__ == '__main__':
    # entry point of the program
    run_linear()


# end of file
