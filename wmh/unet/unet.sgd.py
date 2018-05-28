# a prototypical model for unet neuralogical analysis

from keras.models import Model, load_model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.layers import Conv2DTranspose, BatchNormalization, Dropout, Cropping2D, ZeroPadding2D
from keras.layers import merge
from keras.layers.merge import concatenate
from keras.optimizers import Adam, SGD
from keras import backend as K
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN
from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger

from skimage.transform import AffineTransform
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
    input_shape = (2, IMG_HEIGHT, IMG_WIDTH)
else:
    input_shape = (IMG_HEIGHT, IMG_WIDTH, 2)


def current_time():
    return datetime.now().strftime("%Y-%m-%d_%H:%M:%S")


def get_dirs(bp, paths):
    img_path = [(bp+str(x)+'/pre/FLAIR.nii.gz', bp+str(x)+'/pre/T1.nii.gz') for x in paths]
    mask_path = [bp+str(x)+'/wmh.nii.gz' for x in paths]
    return img_path, mask_path


def get_test_paths():
    bp = '../data/xue/'
    paths = sorted([int(x) for x in os.listdir(bp)])
    test_paths = [x for i, x in enumerate(paths) if i % 4 == 3 and i % 12 != 3]
    test_imgs, test_masks = get_dirs(bp, test_paths)
    return test_imgs, test_masks


def get_paths():
    bp = '../data/xue/'
    paths = sorted([int(x) for x in os.listdir(bp)])
    train_paths = [x for i, x in enumerate(paths) if i % 4 != 3]
    val_paths = [x for i, x in enumerate(paths) if i % 12 == 3]
    train_imgs, train_masks = get_dirs(bp, train_paths)
    val_imgs, val_masks = get_dirs(bp, val_paths)
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
    fmri_shape = (batch, input_shape[0], input_shape[1], input_shape[2])
    mask_shape = (batch, input_shape[0], input_shape[1], 1)

    while True:
        samples = random.sample(range(0, mod), mod)

        for smpl in samples:
            flair = image.load_img(x_path[smpl][0]).get_data()
            flair = normalize(flair)
            t1 = image.load_img(x_path[smpl][1]).get_data()
            t1 = normalize(t1)

            mask = image.load_img(y_path[smpl]).get_data()
            mask[mask == 2.0] = 0.0

            sect_len = flair.shape[2]
            sections = random.sample(range(0, sect_len), sect_len)
            sections = [sections[batch*i: batch*i+batch] for i in range(sect_len//batch)]

            for scts in sections:
                x = np.zeros(shape=fmri_shape, dtype=np.float32)
                y = np.zeros(shape=mask_shape, dtype=np.float32)

                for i, s in enumerate(scts):
                    flair_sect = flair[:, :, s:s+1]
                    t1_sect = t1[:, :, s:s+1]
                    fMRI_sect = np.concatenate((t1_sect, flair_sect), axis=2)
                    mask_sect = mask[:, :, s:s+1]
                    fMRI_sect, mask_sect = pad_img(fMRI_sect, mask_sect)

                    # if transf:
                    #     fMRI_sect, mask_sect = transf(fMRI_sect, mask_sect)

                    x[i] = fMRI_sect
                    y[i] = mask_sect

                yield (x, y)


def get_crop_shape(target, refer):
    # height, the 1 dimension
    # print(K.get_variable_shape(target))
    # print(K.get_variable_shape(refer))

    ch = (K.get_variable_shape(target)[1] - K.get_variable_shape(refer)[1])
    assert (ch >= 0)
    if ch % 2 != 0:
        ch1, ch2 = int(ch / 2), int(ch / 2) + 1
    else:
        ch1, ch2 = int(ch / 2), int(ch / 2)

    cw = (K.get_variable_shape(target)[2] - K.get_variable_shape(refer)[2])
    assert (cw >= 0)
    if cw % 2 != 0:
        cw1, cw2 = int(cw / 2), int(cw / 2) + 1
    else:
        cw1, cw2 = int(cw / 2), int(cw / 2)

    return (ch1, ch2), (cw1, cw2)


def get_pad_shape(target, refer):
    ch = (K.get_variable_shape(refer)[1] - K.get_variable_shape(target)[1])
    assert (ch >= 0)
    if ch % 2 != 0:
        ch1, ch2 = int(ch / 2), int(ch / 2) + 1
    else:
        ch1, ch2 = int(ch / 2), int(ch / 2)

    cw = (K.get_variable_shape(refer)[2] - K.get_variable_shape(target)[2])
    assert (cw >= 0)
    if cw % 2 != 0:
        cw1, cw2 = int(cw / 2), int(cw / 2) + 1
    else:
        cw1, cw2 = int(cw / 2), int(cw / 2)

    return (ch1, ch2), (cw1, cw2)


def create_model(input_shape=(240, 240, 2), bn=True, do=0, ki="he_normal"):
    '''
    bn: if use batchnorm layer
    do: dropout prob
    ki: kernel initializer (glorot_uniform, he_normal, ...)
    lr: learning rate of Adam
    '''
    concat_axis = -1  # the last axis (channel axis)

    inputs = Input(input_shape)  # channels is 2: <t1, flair>

    conv1 = Conv2D(64, (5, 5), padding="same", activation="relu", kernel_initializer=ki)(inputs)
    conv1 = BatchNormalization()(conv1) if bn else conv1
    conv1 = Dropout(do)(conv1) if do else conv1
    conv1 = Conv2D(64, (5, 5), padding="same", activation="relu", kernel_initializer=ki)(conv1)
    conv1 = BatchNormalization()(conv1) if bn else conv1

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(96, (3, 3), padding="same", activation="relu", kernel_initializer=ki)(pool1)
    conv2 = BatchNormalization()(conv2) if bn else conv2
    conv2 = Dropout(do)(conv2) if do else conv2
    conv2 = Conv2D(96, (3, 3), padding="same", activation="relu", kernel_initializer=ki)(conv2)
    conv2 = BatchNormalization()(conv2) if bn else conv2

    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), padding="same", activation="relu", kernel_initializer=ki)(pool2)
    conv3 = BatchNormalization()(conv3) if bn else conv3
    conv3 = Dropout(do)(conv3) if do else conv3
    conv3 = Conv2D(128, (3, 3), padding="same", activation="relu", kernel_initializer=ki)(conv3)
    conv3 = BatchNormalization()(conv3) if bn else conv3

    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), padding="same", activation="relu", kernel_initializer=ki)(pool3)
    conv4 = BatchNormalization()(conv4) if bn else conv4
    conv4 = Dropout(do)(conv4) if do else conv4
    conv4 = Conv2D(256, (3, 3), padding="same", activation="relu", kernel_initializer=ki)(conv4)
    conv4 = BatchNormalization()(conv4) if bn else conv4

    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), padding="same", activation="relu", kernel_initializer=ki)(pool4)
    conv5 = BatchNormalization()(conv5) if bn else conv5
    conv5 = Dropout(do)(conv5) if do else conv5
    conv5 = Conv2D(512, (3, 3), padding="same", activation="relu", kernel_initializer=ki)(conv5)
    conv5 = BatchNormalization()(conv5) if bn else conv5
    upconv5 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same', kernel_initializer=ki)(conv5)

    ch, cw = get_crop_shape(conv4, upconv5)
    crop_conv4 = Cropping2D(cropping=(ch, cw))(conv4)
    cat6 = concatenate([upconv5, crop_conv4], axis=concat_axis)

    conv6 = Conv2D(256, (3, 3), padding="same", activation="relu", kernel_initializer=ki)(cat6)
    conv6 = BatchNormalization()(conv6) if bn else conv6
    conv6 = Dropout(do)(conv6) if do else conv6
    conv6 = Conv2D(256, (3, 3), padding="same", activation="relu", kernel_initializer=ki)(conv6)
    conv6 = BatchNormalization()(conv6) if bn else conv6
    upconv6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same', kernel_initializer=ki)(conv6)

    ch, cw = get_crop_shape(conv3, upconv6)
    crop_conv3 = Cropping2D(cropping=(ch, cw))(conv3)
    up7 = concatenate([upconv6, crop_conv3], axis=concat_axis)

    conv7 = Conv2D(128, (3, 3), padding="same", activation="relu", kernel_initializer=ki)(up7)
    conv7 = BatchNormalization()(conv7) if bn else conv7
    conv7 = Dropout(do)(conv7) if do else conv7
    conv7 = Conv2D(128, (3, 3), padding="same", activation="relu")(conv7)
    conv7 = BatchNormalization()(conv7) if bn else conv7
    upconv7 = Conv2DTranspose(96, (2, 2), strides=(2, 2), padding='same', kernel_initializer=ki)(conv7)

    ch, cw = get_crop_shape(conv2, upconv7)
    crop_conv2 = Cropping2D(cropping=(ch, cw))(conv2)
    up8 = concatenate([upconv7, crop_conv2], axis=concat_axis)

    conv8 = Conv2D(96, (3, 3), padding="same", activation="relu", kernel_initializer=ki)(up8)
    conv8 = BatchNormalization()(conv8) if bn else conv8
    conv8 = Dropout(do)(conv8) if do else conv8
    conv8 = Conv2D(96, (3, 3), padding="same", activation="relu", kernel_initializer=ki)(conv8)
    conv8 = BatchNormalization()(conv8) if bn else conv8
    upconv8 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', kernel_initializer=ki)(conv8)

    ch, cw = get_crop_shape(conv1, upconv8)
    crop_conv1 = Cropping2D(cropping=(ch, cw))(conv1)
    up9 = concatenate([upconv8, crop_conv1], axis=concat_axis)

    conv9 = Conv2D(64, (3, 3), padding="same", activation="relu", kernel_initializer=ki)(up9)
    conv9 = BatchNormalization()(conv9) if bn else conv9
    conv9 = Conv2D(64, (3, 3), padding="same", activation="relu", kernel_initializer=ki)(conv9)
    conv9 = BatchNormalization()(conv9) if bn else conv9
    ch, cw = get_pad_shape(conv9, conv1)
    pad_conv9 = ZeroPadding2D(padding=(ch, cw))(conv9)
    conv9 = Conv2D(1, (1, 1), padding="same", activation="sigmoid", kernel_initializer=ki)(pad_conv9)

    model = Model(inputs=inputs, outputs=conv9)
    return model


def train_model(model, epochs):
    train_imgs, train_masks, val_imgs, val_masks = get_paths()

    steps_per_epoch = get_steps(train_masks)
    validation_steps = get_steps(val_masks)

    attrib = 'xuesgd'
    model_type = 'unet'
    filename = current_time() + '-' + attrib
    os.mkdir('models/' + filename)
    with open('models/' + filename + '/attrib.txt', 'wt') as file_writer:
        file_writer.write('used ' + model_type + '.' + attrib + '.py\n')

    callback_stopping = EarlyStopping(patience=20, monitor='val_loss')
    callback_checkpoint = ModelCheckpoint('models/' + filename + '/model.' + model_type + '.' + attrib +
                                          'epoch_{epoch:02d}.loss_{val_loss:.4f}.hdf5',
                                          monitor='val_loss', save_best_only=True)
    callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                           factor=0.2, patience=5,
                                           verbose=1, min_lr=1e-7)
    callback_csv = CSVLogger('models/' + filename + '/run_log.csv', append=True)
    callback_terminate_nan = TerminateOnNaN()

    model.compile(optimizer=SGD(lr=1e-3, momentum=0.9), loss=dice_coef_loss)

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
    model = create_model(input_shape=input_shape)
    train_model(model, 100)


if __name__ == '__main__':
    run()


# end of file
