# a prototypical model for unet neuralogical analysis


from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN
from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from skimage.transform import AffineTransform
from keras.models import Model, load_model
from keras.optimizers import Adam, SGD
from keras import backend as K
from keras.layers import merge
from skimage import transform
from datetime import datetime
from random import randint
from nilearn import image
import numpy as np
import random
import math
import cv2
import os


BATCH_SIZE = 4

IMG_HEIGHT = 256
IMG_WIDTH = 256


if K.image_data_format() == 'channels_first':
    input_shape = (1, IMG_HEIGHT, IMG_WIDTH)
else:
    input_shape = (IMG_HEIGHT, IMG_WIDTH, 1)


def current_time():
    return datetime.now().strftime("%Y-%m-%d_%H:%M:%S")


def get_dirs(paths):
    fmri_path = ['../data/training/'+str(x)+'/orig/FLAIR.nii.gz' for x in paths]
    mask_path = ['../data/training/'+str(x)+'/wmh.nii.gz' for x in paths]
    return fmri_path, mask_path


def get_test_paths():
    return [x for x in range(4, 60, 4) if x % 12 != 0]


def get_paths():
    train_paths = [x for x in range(0, 60) if x % 4 != 0]
    val_paths = [x for x in range(0, 60, 12)]
    train_imgs, train_masks = get_dirs(train_paths)
    val_imgs, val_masks = get_dirs(val_paths)
    return train_imgs, train_masks, val_imgs, val_masks


def get_steps(x_path):
    slcs = [image.load_img(i).shape[2] for i in x_path]
    return math.ceil(sum(slcs)/BATCH_SIZE)


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def normalize(data):
    xmin, xmax = data.min(), data.max()
    return (data - xmin) / (xmax - xmin)


def pad_img(img, mask):
    hgt, wgt = img.shape[0], img.shape[1]
    hgt_off, wgt_off = (IMG_HEIGHT-hgt)//2, (IMG_WIDTH-wgt)//2
    dimpad = [(hgt_off, hgt_off), (wgt_off, wgt_off), (0, 0)]
    img = np.pad(img, dimpad, 'edge')
    mask = np.pad(mask, dimpad, 'edge')
    return img, mask


def transform_img(img, mask):
    theta = math.radians(randint(-3, 3))
    x_zoom = randint(90, 110)
    y_zoom = randint(-4, 4) + x_zoom
    x_tran = randint(-10, 10) - (x_zoom-100)//2
    y_tran = randint(-10, 10) - (y_zoom-100)//2
    x_zoom /= 100
    y_zoom /= 100

    affine = AffineTransform(scale=(x_zoom, y_zoom),
                             rotation=theta,
                             translation=(x_tran, y_tran))

    img = transform.warp(img, affine.inverse)
    mask = transform.warp(mask, affine.inverse)
    mask[mask >= 0.5] = 1.0

    if random.random() < 0.5:
        img = np.flip(img, 0)
        mask = np.flip(mask, 0)

    return img, mask


def generate_data(x_path, y_path, transf=None):
    mod, batch = len(x_path), BATCH_SIZE
    shape = (batch, IMG_HEIGHT, IMG_WIDTH, 1)

    while True:
        samples = random.sample(range(0, mod), mod)

        for smpl in samples:
            fMRI = image.load_img(x_path[smpl]).get_data()
            fMRI = normalize(fMRI)
            mask = image.load_img(y_path[smpl]).get_data()
            mask[mask == 2.0] = 0.0

            sect_len = fMRI.shape[2]
            sections = random.sample(range(0, sect_len), sect_len)
            sections = [sections[batch*i: batch*i+batch] for i in range(sect_len//batch)]

            for scts in sections:
                x = np.zeros(shape=shape, dtype=np.float32)
                y = np.zeros(shape=shape, dtype=np.float32)

                for i, s in enumerate(scts):
                    fMRI_sect = fMRI[:, :, s:s+1]
                    mask_sect = mask[:, :, s:s+1]
                    fMRI_sect, mask_sect = pad_img(fMRI_sect, mask_sect)

                    if transf:
                        fMRI_sect, mask_sect = transf(fMRI_sect, mask_sect)

                    x[i] = fMRI_sect
                    y[i] = mask_sect

                yield (x, y)


def create_model():

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


def train_model(model, epochs):
    train_imgs, train_masks, val_imgs, val_masks = get_paths()

    steps_per_epoch = get_steps(train_imgs)
    validation_steps = get_steps(val_imgs)

    attrib = 'protomodel'
    model_type = 'unet'
    filename = current_time() + '-' + attrib
    os.mkdir('models/' + filename)
    with open('models/' + filename + '/attrib.txt', 'wt') as file_writer:
        file_writer.write('used ' + model_type + '.' + attrib + '.py\n')

    callback_stopping = EarlyStopping(patience=20, monitor='val_loss')
    callback_checkpoint = ModelCheckpoint('models/' + filename + '/model.' + attrib + '.' + model_type + '.hdf5',
                                          monitor='val_loss', save_best_only=True)
    callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                           factor=0.2, patience=5,
                                           verbose=1, min_lr=1e-7)
    callback_csv = CSVLogger('models/' + filename + '/run_log.csv', append=True)
    callback_terminate_nan = TerminateOnNaN()

    model.compile(optimizer=Adam(lr=1e-4), loss=dice_coef_loss)

    model.fit_generator(generate_data(train_imgs, train_masks, transform_img),
                        steps_per_epoch=steps_per_epoch,
                        epochs=epochs,
                        verbose=1,
                        callbacks=[callback_stopping,
                                   callback_checkpoint,
                                   callback_reduce_lr,
                                   callback_csv,
                                   callback_terminate_nan],
                        validation_data=generate_data(val_imgs, val_masks),
                        validation_steps=validation_steps)

    with open('models/' + filename + '/job_completed.txt', 'wt') as file_writer:
        file_writer.write('job finished gracefully\n')
    model.save('models/' + filename + '/model.' + attrib + '.' + model_type + '.final.hdf5')

    return model


def run():
    model = create_model()
    train_model(model, 100)


if __name__ == '__main__':
    run()


# end of file
