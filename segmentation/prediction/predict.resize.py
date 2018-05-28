# prediction script

from keras.models import load_model
from keras import backend as K
from skimage import transform
from skimage import io
from PIL import Image
import numpy as np
import cv2


MODEL_NAME = 'model.256.unet.40'


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def run_predictions():
    model = load_model(MODEL_NAME+'.hdf5',
                       custom_objects={'dice_coef': dice_coef,
                                       'dice_coef_loss': dice_coef_loss})
    for i in range(1, 928):
        filepath = '../data/testing_images/test_img_'+str(i)+'.jpg'
        img = io.imread(filepath)
        img = np.pad(img, [(3, 3), (3, 3), (0, 0)],  'reflect')

        x = np.array([img])
        prediction = model.predict(x)
        prediction = prediction[0]
        prediction = prediction[3:-3, 3:-3, :] * 255

        prediction[prediction >= 127] = 255
        prediction[prediction < 127] = 0
        cv2.imwrite('test_masks_resize/test_mask_'+str(i)+'.jpg', prediction)
        print('creating mask', i)


def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]  # .T sets Fortran order down-then-right
    run_lengths = []
    prev = -2
    for b in dots:
        if b > prev + 1:
            run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


def gen_encodings():
    masks = ['test_masks_resize/test_mask_'+str(x)+'.jpg' for x in range(1, 928)]
    encodings = []
    for i, m in enumerate(masks):
        img = Image.open(m)
        x = np.array(img.getdata(), dtype=np.uint8).reshape(img.size[::-1])
        x = x // 255
        encodings.append(rle_encoding(x))
        print('creating encoding', i+1)

    conv = lambda l: ' '.join(map(str, l))  # list -> string
    with open('encodings.'+MODEL_NAME+'.csv', 'wt') as file_writer:
        file_writer.write('ImageId,EncodedPixels\n')
        for i, encoding in enumerate(encodings):
            entry = conv(encoding)
            file_writer.write('test_mask_'+str(i+1)+','+entry+'\n')


if __name__ == '__main__':
    run_predictions()
    gen_encodings()

