# from keras.applications import VGG16

from skimage import color
from skimage import util
from skimage import io
import numpy as np
import dlib
import os


detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor()
win = dlib.image_window()


def show_dlib(img, overlay=None):
    win.clear_overlay()
    win.set_image(img)
    if overlay:
        print(overlay)
        win.add_overlay(overlay)
    dlib.hit_enter_to_continue()


def process_img(img):
    # TODO determine if cropping is advisable, and to what degree
    # TODO perhaps rotate face image for straight ahead look?
    det = detector(img, 1)
    if len(det) != 1:
        return None
    det = det[0]
    btm, top, rgt, lft = det.bottom(), det.top(), det.right(), det.left()
    wth, hgt = (rgt-lft)//4, (btm-top)//4
    img = img[top-hgt:btm+hgt, lft-wth:rgt+wth, ]
    show_dlib(img)
    print(img.shape)
    if len(img.shape) == 2:
        img = color.gray2rgb(img)
    return img


def import_imgs():
    bp = 'data/train'
    paths = [bp+'/'+x for x in os.listdir(bp)]

    # TODO remove later
    paths = paths[90:100]

    imgs = io.imread_collection(paths)
    proc_imgs = {}
    for path, img in zip(paths, imgs):
        img = process_img(img)
        if img is not None:
            proc_imgs[path] = {}
            proc_imgs[path]['img'] = img

    # TODO convert to np array
    return proc_imgs


def run():
    imgs = import_imgs()



run()

