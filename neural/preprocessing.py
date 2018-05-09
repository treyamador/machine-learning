# from keras.applications import VGG16

from skimage import transform
from skimage import color
from skimage import io
import dlib
import cv2
import os


BORDER_NORM = 2
DEFAULT_DIMENSION = 224


detector = dlib.get_frontal_face_detector()
# win = dlib.image_window()


def show_cv(img):
    cv2.imshow('', img)
    cv2.waitKey(0)


def read_training():
    training = {}
    with open('data/train_target.csv', 'rt') as f_obj:
        for line in f_obj.readlines()[1:]:
            key, val = line.split(',')
            training[key] = int(val)
    return training


'''

def process_img(img):
    img = transform.resize(img, (DEFAULT_DIMENSION, DEFAULT_DIMENSION,))
    if len(img.shape) == 2:
        img = color.gray2rgb(img)
    return img


'''


def process_img(img):
    # TODO determine if cropping is advisable, and to what degree
    # TODO perhaps rotate face image for straight ahead look?
    try:
        det = detector(img, 1)
    except RuntimeError as err:
        print(err)
        return None
    if len(det) != 1:
        return None

    det = det[0]
    btm, top, rgt, lft = det.bottom(), det.top(), det.right(), det.left()
    wth, hgt = (rgt-lft)//BORDER_NORM, (btm-top)//BORDER_NORM
    top_off, btm_off, lft_off, rgt_off = top-hgt, btm+hgt, lft-wth, rgt+wth

    # shifting offset up
    # top_off -= hgt
    # btm_off -= hgt

    top_off = top_off if top_off > 0 else 0
    btm_off = btm_off if btm_off < img.shape[0] else img.shape[0]
    lft_off = lft_off if lft_off > 0 else 0
    rgt_off = rgt_off if rgt_off < img.shape[1] else img.shape[1]

    hgt_off = btm_off - top_off
    wgt_off = rgt_off - lft_off
    if hgt_off < DEFAULT_DIMENSION // 2 or wgt_off < DEFAULT_DIMENSION // 2:
        return None

    # if hgt_off != wgt_off:
    #     return None

    if hgt_off < 3*(wgt_off//4):
        return None
    if wgt_off < 3*(hgt_off//4):
        return None

    img = img[top_off:btm_off, lft_off:rgt_off, ]

    # TODO remove this, or write it programmatically
    img = transform.resize(img, (DEFAULT_DIMENSION, DEFAULT_DIMENSION,))

    if len(img.shape) == 2:
        img = color.gray2rgb(img)
    return img


def get_dir():
    return 'data/modtrain'+'-d'+str(DEFAULT_DIMENSION)+''


def run():
    bp = 'data/train'
    mp = get_dir()
    os.mkdir(mp)
    exts = [x for x in os.listdir(bp)]
    paths = [bp+'/'+x for x in exts]
    directory = read_training()
    imgs = io.imread_collection(paths)
    ttl = len(exts)
    with open(mp+'_target.csv', 'wt') as f_obj:
        itr = 1
        for ext, img in zip(exts, imgs):
            img = process_img(img)
            f_obj.write(ext+','+str(directory[ext])+'\n')
            io.imsave(mp+'/'+ext, img)
            print(itr, 'of', ttl)
            itr += 1


if __name__ == '__main__':
    run()