# unet hsv

from keras.models import Model, load_model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, merge, Dropout
from keras.layers.core import SpatialDropout2D, Activation
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, TerminateOnNaN, ReduceLROnPlateau
from keras.optimizers import Adam, SGD, RMSprop
import numpy as np
from skimage import transform
from skimage.transform import AffineTransform
from scipy import ndimage
from datetime import datetime
from random import randint
import random
from math import ceil
import cv2
import os


BATCH_SIZE = 4

IMG_HEIGHT = 224
IMG_WIDTH = 224

# Number of image channels (for example 3 in case of RGB, or 1 for grayscale images)
INPUT_CHANNELS = 3
# Number of output masks (1 in case you predict only one type of objects)
OUTPUT_MASK_CHANNELS = 1


if K.image_data_format() == 'channels_first':
    input_shape = (3, IMG_WIDTH, IMG_HEIGHT)
else:
    input_shape = (IMG_WIDTH, IMG_HEIGHT, 3)


def get_train_imgs():
    return ['../data/training/images/train_img_'+str(x)+'.jpg' for x in range(1, 1501)]


def get_train_masks():
    return ['../data/training/masks/train_mask_'+str(x)+'.ppm' for x in range(1, 1501)]


def get_val_imgs():
    return ['../data/validation/images/validation_img_'+str(x)+'.jpg' for x in range(1, 501)]


def get_val_masks():
    return ['../data/validation/masks/validation_mask_'+str(x)+'.ppm' for x in range(1, 501)]


def current_time():
    return datetime.now().strftime("%Y-%m-%d_%H:%M:%S")


def get_img_mask():
    idx = 300
    train_imgs = get_train_imgs()
    train_masks = get_train_masks()
    train_imgs.extend(get_val_imgs()[:idx])
    train_masks.extend(get_val_masks()[:idx])
    val_imgs = get_val_imgs()[idx:]
    val_masks = get_val_masks()[idx:]
    return train_imgs, train_masks, val_imgs, val_masks


def random_hue_shift(image, hue_shift_limit=(-10, 10),
                     sat_shift_limit=(-20, 20),
                     val_shift_limit=(-20, 20), u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.uniform(hue_shift_limit[0], hue_shift_limit[1])
        h = cv2.add(h, hue_shift)
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    return image


def transform_img(img, mask):
    theta = randint(-100, 100) / 1800
    zoom = randint(1000, 1100) / 1000
    shear = randint(-100, 100) / 1800
    hrz_trans = randint(-20, 20)
    vrt_trans = randint(-20, 20)

    affine = AffineTransform(scale=(zoom, zoom),
                             rotation=theta,
                             shear=shear,
                             translation=(hrz_trans, vrt_trans))

    img = transform.warp(img, affine.inverse)
    mask = transform.warp(mask, affine.inverse)

    if random.random() < 0.5:
        img = np.flip(img, 1)
        mask = np.flip(mask, 1)

    return img, mask


def generate_data(x_path, y_path, batch):
    mod, idx = len(x_path), 0
    img_shape = (batch, input_shape[0], input_shape[1], input_shape[2])
    mask_shape = (batch, input_shape[0], input_shape[1], 1)

    while True:
        samples = random.sample(range(0, mod), mod)
        samples = [samples[batch*i: batch*i+batch] for i in range(mod//batch)]

        for smpl in samples:
            x = np.zeros(shape=img_shape, dtype=np.float32)
            y = np.zeros(shape=mask_shape, dtype=np.float32)

            for i, s in enumerate(smpl):
                img, msk = ndimage.imread(x_path[s]), ndimage.imread(y_path[s])
                msk = msk[:, :, 1:2]
                img, msk = transform_img(img, msk)
                x[i], y[i] = img, msk

            yield (x, y)


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def create_unet():

    inputs = Input(input_shape)

    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    # drop4 = Dropout(0.3)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    # drop5 = Dropout(0.3)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv5))
    merge6 = merge([conv4, up6], mode='concat', concat_axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = merge([conv3, up7], mode='concat', concat_axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = merge([conv2, up8], mode='concat', concat_axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = merge([conv1, up9], mode='concat', concat_axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    return model


def train_unet(model, epochs):
    train_imgs, train_masks, val_imgs, val_masks = get_img_mask()

    steps_per_epoch = int(ceil(len(train_imgs) / BATCH_SIZE))
    validation_steps = int(ceil(len(val_imgs) / BATCH_SIZE))

    attrib = 'lowsgd'
    model_type = 'unet'
    filename = current_time()+'-'+attrib
    os.mkdir('models/'+filename)

    with open('models/'+filename+'/attrib.txt', 'wt') as file_writer:
        file_writer.write('used '+model_type+'.'+attrib+'.py\n')

    callback_stopping = EarlyStopping(patience=25, monitor='val_loss')
    callback_checkpoint = ModelCheckpoint('models/' + filename + '/model.'+attrib+'.'+model_type+'.hdf5',
                                          monitor='val_loss', save_best_only=True)
    callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                           factor=0.2, patience=5,
                                           verbose=1, min_lr=1e-7)
    callback_csv = CSVLogger('models/' + filename + '/run_log.csv', append=True)
    callback_terminate_nan = TerminateOnNaN()

    model.compile(optimizer=SGD(lr=1e-5, momentum=0.9), loss=dice_coef_loss, metrics=[dice_coef])

    model.fit_generator(generate_data(train_imgs, train_masks, BATCH_SIZE),
                        steps_per_epoch=steps_per_epoch,
                        epochs=epochs,
                        verbose=1,
                        callbacks=[callback_stopping,
                                   callback_checkpoint,
                                   callback_reduce_lr,
                                   callback_csv,
                                   callback_terminate_nan],
                        validation_data=generate_data(val_imgs, val_masks, BATCH_SIZE),
                        validation_steps=validation_steps)

    with open('models/'+filename+'/job_completed.txt', 'wt') as file_writer:
        file_writer.write('job finished gracefully\n')
    model.save('models/' + filename + '/model.'+attrib+'.'+model_type+'.final.hdf5')

    return model


def run():
    model = load_model('models/2018-05-19_07:55:35-nowdrop/model.nowdrop.unet.98.hdf5',
                       custom_objects={'dice_coef': dice_coef,
                                       'dice_coef_loss': dice_coef_loss})
    train_unet(model, 100)


if __name__ == '__main__':
    run()


# end of file
