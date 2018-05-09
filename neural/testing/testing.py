from keras.models import load_model
from keras.preprocessing import image
from skimage import io, transform
import numpy as np
from math import ceil
import cv2


DIMENSION = 224
BATCH_SIZE = 32


def get_paths():
    init, end = 1, 7090
    return ['test_'+str(x)+'.jpg' for x in range(init, end+1)]


def run():
    paths = get_paths()
    batches = int(ceil(len(paths)/BATCH_SIZE))
    name = 'model.vgg16.30.loss-0.21.hdf5'
    model = load_model(name)
    with open('vgg16.30-0.21.cnn.csv', 'wt') as file_writer:
        file_writer.write('Id,Expected\n')
        for i in range(batches):
            pth = paths[BATCH_SIZE*i:BATCH_SIZE*i+BATCH_SIZE]
            X = np.array([transform.resize(io.imread('test/'+p), (DIMENSION, DIMENSION, 3)) for p in pth])
            # predictions = np.round(model.predict(X))
            predictions = model.predict(X)

            for j in range(predictions.shape[0]):
                # age = int(predictions[j][0])
                age = predictions[j][0]
                file_writer.write(pth[j]+','+str(age)+'\n')
                print(pth[j]+' age '+str(age))
            # print('step', i, 'of', batches)


if __name__ == '__main__':
    run()

