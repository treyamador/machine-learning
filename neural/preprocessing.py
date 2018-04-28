# from keras.applications import VGG16

from skimage import transform
from skimage import color
from skimage import io
import dlib
import cv2
import os


detector = dlib.get_frontal_face_detector()
win = dlib.image_window()


def show_dlib(img, overlay=None):
    win.clear_overlay()
    win.set_image(img)
    if overlay:
        print(overlay)
        win.add_overlay(overlay)
    dlib.hit_enter_to_continue()


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


def process_img(img):
    # TODO determine if cropping is advisable, and to what degree
    # TODO perhaps rotate face image for straight ahead look?
    det = detector(img, 1)
    if len(det) != 1:
        return None

    det = det[0]
    btm, top, rgt, lft = det.bottom(), det.top(), det.right(), det.left()
    wth, hgt = (rgt-lft)//3, (btm-top)//3
    top_off, btm_off, lft_off, rgt_off = top-hgt, btm+hgt, lft-wth, rgt+wth

    top_off = top_off if top_off > 0 else 0
    btm_off = btm_off if btm_off < img.shape[0] else img.shape[0]
    lft_off = lft_off if lft_off > 0 else 0
    rgt_off = rgt_off if rgt_off < img.shape[1] else img.shape[1]

    img = img[top_off:btm_off, lft_off:rgt_off, ]
    img = transform.resize(img, (500, 500,))

    if len(img.shape) == 2:
        img = color.gray2rgb(img)
    return img


def import_imgs():
    bp = 'data/train'
    mp = 'data/modtrain'
    exts = [x for x in os.listdir(bp)]
    paths = [bp+'/'+x for x in exts]
    directory = read_training()
    imgs = io.imread_collection(paths)
    with open('data/modtrain_target.csv', 'wt') as f_obj:
        for ext, img in zip(exts, imgs):
            img = process_img(img)
            if img is not None:
                f_obj.write(ext+','+str(directory[ext])+'\n')
                io.imsave(mp+'/'+ext, img)


def run():
    import_imgs()


if __name__ == '__main__':
    run()

