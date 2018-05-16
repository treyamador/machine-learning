from unet.unetzf import ZF_UNET_224
import numpy as np
from skimage import transform
from skimage.transform import AffineTransform
from skimage import io
import cv2
from random import randint, random

import math


def get_train_imgs():
    return ['../data/training_hair_skin/images/train_img_'+str(x)+'.jpg' for x in range(1, 1501)]


def get_train_masks():
    return ['../data/training_hair_skin/masks/train_mask_'+str(x)+'.ppm' for x in range(1, 1501)]


def get_val_imgs():
    return ['../data/validation_hair_skin/images/validation_img_'+str(x)+'.jpg' for x in range(1, 501)]


def get_val_masks():
    return ['../data/validation_hair_skin/masks/validation_mask_'+str(x)+'.ppm' for x in range(1, 501)]


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

    if random() < 0.5:
        img = np.flip(img, 1)
        mask = np.flip(mask, 1)

    return img, mask


def run_unet():
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

