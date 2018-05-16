from unet.unetzf import ZF_UNET_224, dice_coef_loss, dice_coef
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, TerminateOnNaN
from keras.optimizers import Adam, SGD
import numpy as np
from skimage import transform
from skimage.transform import AffineTransform
from scipy import ndimage
from datetime import datetime
import cv2
from random import randint
import random
import os

from keras import backend as K


BATCH_SIZE = 100
# PIXEL_NORMAL = 255.0

IMG_HEIGHT = 224
IMG_WIDTH = 224

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


def transform_img(img, mask):
    theta = randint(-100, 100) / 1800
    zoom = randint(950, 1050) / 1000
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


# TODO check if all validation images are used in batch
def generate_data(x_path, y_path, batch_size):
    mod = len(x_path)
    shape = (batch_size, input_shape[0], input_shape[1], input_shape[2])
    while True:
        smpl = random.sample(range(0, mod), batch_size)
        x = np.zeros(shape=shape, dtype=np.float32)
        y = np.zeros(shape=shape, dtype=np.float32)
        for i, s in enumerate(smpl):
            img, msk = ndimage.imread(x_path[s]), ndimage.imread(y_path[s])
            img, msk = transform_img(img, msk)
            msk[msk > 0.5] = 1
            msk[msk <= 0.5] = 0
            x[i], y[i] = img, msk
        yield (x, y)


def train_unet(model, epochs):
    train_imgs, train_masks, val_imgs, val_masks = get_img_mask()

    steps_per_epoch = len(train_imgs) / BATCH_SIZE
    validation_steps = len(val_imgs) / len(val_imgs)

    attrib = 'untrained'
    model_type = 'unet'
    filename = current_time()+'-'+attrib
    os.mkdir('models/'+filename)

    with open('models/'+filename+'/attrib.txt', 'wt') as file_writer:
        file_writer.write('used '+model_type+'.'+attrib+'.py\n')

    callback_stopping = EarlyStopping(patience=20, monitor='val_loss')
    callback_checkpoint = ModelCheckpoint('models/' + filename + '/model.'+attrib+'.'+model_type+'.hdf5',
                                          monitor='val_loss', save_best_only=True)
    callback_csv = CSVLogger('models/' + filename + '/run_log.csv', append=True)
    callback_terminate_nan = TerminateOnNaN()

    model.compile(optimizer=Adam(lr=0.0001),
                  loss=dice_coef_loss,
                  metrics=[dice_coef])

    model.fit_generator(generate_data(train_imgs, train_masks, BATCH_SIZE),
                        steps_per_epoch=steps_per_epoch,
                        epochs=epochs,
                        verbose=1,
                        callbacks=[callback_checkpoint,
                                   callback_stopping,
                                   callback_csv,
                                   callback_terminate_nan],
                        validation_data=generate_data(val_imgs, val_masks, len(val_imgs)),
                        validation_steps=validation_steps)

    with open('models/'+filename+'/job_completed.txt', 'wt') as file_writer:
        file_writer.write('job finished gracefully\n')

    return model


def run():
    model = ZF_UNET_224()
    train_unet(model, 100)


if __name__ == '__main__':
    run()


