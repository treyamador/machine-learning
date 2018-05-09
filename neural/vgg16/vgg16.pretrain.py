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

# base path to image files
BASE_IMG_PATH = '../data/modtrain-d224'

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

def sep_paths():
    # function splits path to image and ages of target
    # as read by a file at the base path
    with open(BASE_IMG_PATH+'_target.csv') as f_obj:
        # init empty arrays for image paths and ages
        paths, ages = [], []
        # iterate through each line in the filereader object
        for line in f_obj:
            # retrieve the path to image and age of subject
            path, age = line.split(',')
            # convert str age to int age
            age = int(age)
            # add path to filepath array
            paths.append(path)
            # add age to age array
            ages.append(age)
    # create array with basepath to directory of images
    paths = [BASE_IMG_PATH+'/'+x for x in paths]
    # create index to split 1/5 for validation images
    i = (4*len(paths))//5
    # splice training and validation images into distinct categories
    train_path, val_path = paths[:i], paths[i:]
    # splice training and validation ages into distinct categories
    train_age, val_age = ages[:i], ages[i:]
    # return the training and validation data and targets
    return train_path, train_age, val_path, val_age


def generate_data(x_paths, y_target, batch):
    # a generator function to feed the data into the model training function
    # prevents the memory from having to be loaded in at once
    # get total number of data points
    mod = len(x_paths)
    # iterate in perpetuity for the generator,
    # as specified by Keras documentation
    while True:
        # create a series of random, non-repeating indices
        # for which to draw samples from the training and validation data
        s = random.sample(range(0, mod), batch)
        # return the images at the specified indices
        imgs = [ndimage.imread(x_paths[i]) for i in s]
        # return the ages at the specified indices
        trgs = [y_target[i] for i in s]
        # convert into an np array of type float
        # with the pixel information corrected
        # to be between 0 and 1
        x = np.array(imgs, dtype=np.float32) / PIXEL_NORMAL
        # convert age info into float
        y = np.array(trgs, dtype=np.float32)
        # yeild the data and target to model with generator
        yield (x, y)


def current_time():
    # return the current time to the seconds as a string
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

    # collect train and validation data and targets
    train_paths, train_ages, val_paths, val_ages = sep_paths()
    # count the number of samples per epoch in the training set
    steps_per_epoch = len(train_paths) / BATCH_SIZE
    # count the number of samples per epoch in the validation set
    validation_steps = len(val_paths) / VALID_BATCH
    # assign attribute, a description of the paradigm being tested
    attrib = 'adam-pretrain'
    # specify the neural net architecture
    model_type = 'vgg16'
    # create a directory path to allow placement of items
    file_time = current_time()+'-'+attrib
    # make a directory at specified path for this experiment
    os.mkdir('models/'+file_time)
    # open a new file to write some initial information about experiment
    with open('models/'+file_time+'/attrib.txt', 'wt') as file_writer:
        # write the spefied information to a text file
        file_writer.write('used '+model_type+'.'+attrib+'.py\n')

    # early stopping callback prevents model from running when
    #   results have stopped inproving
    # callback_stopping = EarlyStopping(patience=10, monitor='val_loss')

    # model checkpoint saves the model, which gives users the ability to later load the model into memory
    callback_checkpoint = ModelCheckpoint('models/'+file_time+'/model.' + model_type + '-' + attrib +
                                          '.{epoch:02d}.mse-{val_loss:.2f}.hdf5',
                                          monitor='val_loss')

    # the csv logger accounts for loss data and accuracy
    callback_csv = CSVLogger('models/'+file_time+'/run_log.csv', append=True)
    # create callback to terminate when a loss value is encountered
    callback_terminate_nan = TerminateOnNaN()
    # tensorboard allows user to observe trends in data as they are calculated
    callback_tensor_board = TensorBoard(log_dir='./Graph', histogram_freq=0,
                                        write_graph=True, write_images=True)
    # train the model
    #   a user-defined train generator feeds images and target data to model
    #   epochs is the number of epochs used
    #   step sizes are the number of images to step
    #   the callback array is passed in
    #   the validation generator feeds validation images to train model
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

    # write to a file upon the successful completion of the task at hand
    with open('models/'+file_time+'/job_completed.txt', 'wt') as file_writer:
        # write a simple text message to a file
        file_writer.write('job finished gracefully\n')
    # return the trained model
    return model


def run_linear():
    # run the model training and fitting
    # create a new Keras model, previously trained with Imagenet weights 
    # model = create_VGG16()
    # load a model trained by the user
    model = load_model('models/2018-05-07_10:08:19-pretraining/model.vgg16-pretraining.09.mse-257.17.hdf5')

    # a model such as below would be used to train only the top layer
    '''
    
    for layer in model.layers[:-6]:
        layer.trainable = False
    for layer in model.layers[-6:]:
        layer.trainable = True

    model.compile(loss='mse',
                  optimizer='rmsprop',
                  metrics=['acc', 'mse'])
    model = train_VGG16(model, 10)
    
    '''

    # a model such as below would be used to train
    # both the top layer (including pooling and dense layers)
    # as well as the top convolutional layers
    # iterate through each layer
    for layer in model.layers[:-10]:
        # set bottom layers to untrainable
        layer.trainable = False
    # iterate through top layers
    for layer in model.layers[-10:]:
        # set top layers to trainable
        layer.trainable = True
    # compile the model
    # during the preliminary training, only the new layers are trainable
    #   the optimizer used was typically Adam
    #   However, during convolutional layer training,
    #   a lower learning rate is required such that 
    #   the weights of the convolutional layer are not skewed
    #   mean squared error is the loss function used in these trials
    model.compile(loss='mse',
                  optimizer=Adam(),
                  metrics=['acc', 'mse'])
    # the model is trained for a specified number of epochs
    train_VGG16(model, 100)


if __name__ == '__main__':
    # entry point of the program
    run_linear()


# end of file
