from unet.unetzf import ZF_UNET_224
import numpy as np
from skimage import transform
from skimage.transform import AffineTransform
from skimage import io
from datetime import datetime
import cv2
from random import randint
import random

from keras import backend as K



BATCH_SIZE = 100
PIXEL_NORMAL = 255.0

IMG_HEIGHT = 250
IMG_WIDTH = 250

if K.image_data_format() == 'channels_first':
    input_shape = (3, IMG_WIDTH, IMG_HEIGHT)
else:
    input_shape = (IMG_WIDTH, IMG_HEIGHT, 3)


def get_train_imgs():
    return ['../data/training_hair_skin/images/train_img_'+str(x)+'.jpg' for x in range(1, 1501)]


def get_train_masks():
    return ['../data/training_hair_skin/masks/train_mask_'+str(x)+'.ppm' for x in range(1, 1501)]


def get_val_imgs():
    return ['../data/validation_hair_skin/images/validation_img_'+str(x)+'.jpg' for x in range(1, 501)]


def get_val_masks():
    return ['../data/validation_hair_skin/masks/validation_mask_'+str(x)+'.ppm' for x in range(1, 501)]


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

    hrz_trans = randint(-5, 5)
    vrt_trans = randint(-5, 5)
    hrz_trans = (-zoom * hrz_trans) + hrz_trans
    vrt_trans = (-zoom * vrt_trans) + vrt_trans

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


def generate_data(x_path, y_path):
    mod = len(x_path)
    shape = (BATCH_SIZE, input_shape[0], input_shape[1], input_shape[2])
    while True:
        smpl = random.sample(range(0, mod), BATCH_SIZE)
        x = np.zeros(shape=shape, dtype=np.float32)
        y = np.zeros(shape=shape, dtype=np.float32)
        for i, s in enumerate(smpl):
            img, msk = io.imread(x_path[s]), io.imread(y_path[s])
            img, msk = transform_img(img, msk)
            x[i], y[i] = img, msk
        yield (x, y)


def run_unet():
    train_imgs, train_masks, val_imgs, val_masks = get_img_mask()

    x, y = generate_data(train_imgs, train_masks)

    cv2.imshow('', x[0])
    cv2.waitKey(0)
    cv2.imshow('', y[0])
    cv2.waitKey(0)


def test_unet():
    # model = ZF_UNET_224()
    # model.summary()
    train_imgs = get_train_imgs()
    train_masks = get_train_masks()
    idx = 0
    for img_path, mask_path in zip(train_imgs, train_masks):
        img = io.imread(img_path)
        mask = io.imread(mask_path)
        img, mask = transform_img(img, mask)
        print(img.shape)
        cv2.imshow('', img)
        cv2.waitKey(0)
        cv2.imshow('', mask)
        cv2.waitKey(0)
        idx += 1
        if idx > 10:
            break


if __name__ == '__main__':
    run_unet()

