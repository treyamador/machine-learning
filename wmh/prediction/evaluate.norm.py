# predicts masks


from keras.models import load_model
from keras.optimizers import Adam
from keras import backend as K
from nilearn import image
import numpy as np
import math
import cv2
import os


IMG_HEIGHT = 240
IMG_WIDTH = 240

MODEL_NAME = 'models/model.unet.adamlinearepoch_52.loss_-0.7610.hdf5'


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


def get_section():
    img_paths, mask_paths = get_test_paths()

    flair = image.load_img(img_paths[0][0]).get_data()
    t1 = image.load_img(img_paths[0][1]).get_data()

    hlf = (flair.shape[2]//2)
    flair_sect = flair[:, :, hlf:hlf+1]
    t1_sect = t1[:, :, hlf:hlf+1]
    fMRI_sect = np.concatenate((t1_sect, flair_sect), axis=2)
    fMRI_sect = np.pad(fMRI_sect, [(8, 8), (8, 8), (0, 0)], 'edge')

    mask = image.load_img(mask_paths[0]).get_data()
    mask_sect = mask[:, :, hlf:hlf+1]
    mask_sect = np.pad(mask_sect, [(8, 8), (8, 8), (0, 0)], 'edge')

    return fMRI_sect, mask_sect


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def get_slices(x_path):
    slcs = 0
    for path in x_path:
        slc_len = image.load_img(path).shape[2]
        slcs += slc_len
        print(slc_len, sep=', ', end=', ')
    print('')
    return slcs


def normalize(data):
    xmin, xmax = data.min(), data.max()
    if xmax == xmin:
        return data
    return (data - xmin) / (xmax - xmin)


def partition(imgs, idx=8):
    img_len = imgs.shape[2]
    hgt, wgt = imgs.shape[0], imgs.shape[1]
    li, lf = img_len//idx, ((idx-1)*img_len)//idx
    hi, hf = (hgt-IMG_HEIGHT, IMG_HEIGHT-hgt) if hgt-IMG_HEIGHT > 0 else (0, IMG_HEIGHT)
    wi, wf = (wgt-IMG_WIDTH, IMG_WIDTH-wgt) if wgt-IMG_WIDTH > 0 else (0, IMG_WIDTH)
    return imgs[hi:hf, wi:wf, :]


def pad_img(img, mask):
    hgt, wgt = img.shape[0], img.shape[1]
    hgt_off, wgt_off = (IMG_HEIGHT-hgt)//2, (IMG_WIDTH-wgt)//2
    dimpad = [(hgt_off, hgt_off), (wgt_off, wgt_off), (0, 0)]
    img = np.pad(img, dimpad, 'edge')
    mask = np.pad(mask, dimpad, 'edge')
    return img, mask


def load_data():
    test_imgs, test_masks = get_test_paths()
    slcs = get_slices(test_masks)

    fMRIs = np.zeros((slcs, IMG_HEIGHT, IMG_WIDTH, 2))
    masks = np.zeros((slcs, IMG_HEIGHT, IMG_WIDTH, 1))

    idx = 0

    for img, msk in zip(test_imgs, test_masks):
        flair = image.load_img(img[0]).get_data()
        t1 = image.load_img(img[1]).get_data()
        mask = image.load_img(msk).get_data()

        flair = partition(flair)
        t1 = partition(t1)
        mask = partition(mask)

        mask[mask == 2.0] = 0.0

        for i in range(flair.shape[2]):
            flair_sect = normalize(flair[:, :, i])
            t1_sect = normalize(t1[:, :, i])
            fMRI_sect = np.stack((t1_sect, flair_sect), axis=-1)
            mask_sect = mask[:, :, i:i + 1]

            fMRI_sect, mask_sect = pad_img(fMRI_sect, mask_sect)

            fMRIs[idx] = fMRI_sect
            masks[idx] = mask_sect

            idx += 1

    return fMRIs, masks


def evaluate_model():
    fMRIs, masks = load_data()
    print(fMRIs.shape)
    print(masks.shape)

    model = load_model(MODEL_NAME,
                       custom_objects={'dice_coef': dice_coef,
                                       'dice_coef_loss': dice_coef_loss})
    scores = model.evaluate(fMRIs, masks, batch_size=4, verbose=1)

    print(scores)


def predict_model():
    fMRIs, masks = load_data()
    idx = 6*48+44

    model = load_model(MODEL_NAME,
                       custom_objects={'dice_coef': dice_coef,
                                       'dice_coef_loss': dice_coef_loss})

    x = fMRIs[idx:idx+1, :, :, :]
    pred = model.predict(x, batch_size=4)
    grnd = masks[idx:idx+1, :, :, :]

    for i in range(x.shape[0]):
        print('mask', i+1)
        cv2.imshow('', grnd[i])
        cv2.waitKey(0)
        print('pred', i+1)
        cv2.imshow('', pred[i])
        cv2.waitKey(0)


if __name__ == '__main__':
    evaluate_model()
    # predict_model()

