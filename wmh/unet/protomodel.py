# a prototypical model for unet neuralogical analysis

from skimage.transform import AffineTransform
from keras import backend as K
from skimage import transform
from datetime import datetime
from random import randint
import numpy as np
import nilearn
import random
import math
import cv2


BATCH_SIZE = 4

IMG_HEIGHT = 256
IMG_WIDTH = 256


if K.image_data_format() == 'channels_first':
    input_shape = (1, IMG_WIDTH, IMG_HEIGHT)
else:
    input_shape = (IMG_WIDTH, IMG_HEIGHT, 1)


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


def generate_data(x_path, y_path, transform=None):
    mod, batch = len(x_path), BATCH_SIZE
    shape = (batch, IMG_HEIGHT, IMG_WIDTH, 1)

    while True:
        samples = random.sample(range(0, mod), mod)

        for smpl in samples:
            fMRI = nilearn.image.load_img(x_path[smpl]).get_data()
            fMRI = normalize(fMRI)
            mask = nilearn.image.load_img(y_path[smpl]).get_data()
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

                    print(fMRI_sect.min(), fMRI_sect.max())
                    print(mask_sect.min(), mask_sect.max())
                    cv2.imshow('', fMRI_sect)
                    cv2.waitKey(0)
                    # cv2.imshow('', mask_sect)
                    # cv2.waitKey(0)

                    if transform:
                        fMRI_sect, mask_sect = transform(fMRI_sect, mask_sect)

                    print(fMRI_sect.min(), fMRI_sect.max())
                    print(mask_sect.min(), mask_sect.max())
                    cv2.imshow('', fMRI_sect)
                    cv2.waitKey(0)
                    # cv2.imshow('', mask_sect)
                    # cv2.waitKey(0)

                    print('\n')

                    x[i] = fMRI_sect
                    y[i] = mask_sect

                # TODO uncomment
                # yield (x, y)


def create_model():
    return 0


def train_model(model, epochs):
    train_imgs, train_paths, val_imgs, val_paths = get_paths()

    # TODO remove generator
    generate_data(train_imgs, train_paths, transform_img)


def run():
    model = create_model()
    train_model(model, 100)


if __name__ == '__main__':
    run()


# end of file
