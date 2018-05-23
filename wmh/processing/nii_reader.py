# read nii images
from nilearn import image
from nilearn import plotting
import pylab
import cv2


def normalize(data):
    xmin, xmax = data.min(), data.max()
    return (data - xmin) / (xmax - xmin)


def read():
    mripath = '../data/training/57/orig/FLAIR.nii.gz'
    maskpath = '../data/training/57/wmh.nii.gz'
    fMRI = image.load_img(mripath)
    mask = image.load_img(maskpath)

    print(fMRI.shape)
    print(mask.shape)

    preslc, sufslc = '../data/ztest/', '_slice.tiff'
    premsk, sufmsk = '../data/ztest/', '_mask.tiff'

    for i in range(fMRI.shape[2]):
        pylab.imsave(preslc+str(i)+sufslc, fMRI.get_data()[:, :, i])
        pylab.imsave(premsk+str(i)+sufmsk, mask.get_data()[:, :, i])

    data = fMRI.get_data()
    # data = mask.get_data()

    slc_num = 40

    slc = data[:, :, slc_num]
    print(slc[70])
    print(data.min(), data.max())
    print(data.dtype)
    print('\n\n')

    data = normalize(data)

    slc = data[:, :, slc_num]
    print(slc[70])
    print(data.min(), data.max())
    print(data.dtype)

    cv2.imshow('', slc)
    cv2.waitKey(0)


if __name__ == '__main__':
    read()

