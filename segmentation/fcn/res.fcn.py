
from KerasFCN.models import FCN_Vgg16_32s


# unet weighted

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout, InputSpec, Layer
from keras.layers import concatenate, Conv2DTranspose, merge
from keras.layers.normalization import BatchNormalization
from keras.layers.core import SpatialDropout2D, Activation
from keras.layers.merge import Add
from keras.regularizers import l2
from keras import backend as K
from keras.backend import permute_dimensions
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, TerminateOnNaN, ReduceLROnPlateau
from keras.optimizers import Adam, SGD, RMSprop
import tensorflow as tf
import numpy as np
from skimage import transform
from skimage.transform import AffineTransform
from scipy import ndimage
from datetime import datetime
from random import randint
import random
from math import ceil
import os


BATCH_SIZE = 4

IMG_HEIGHT = 224
IMG_WIDTH = 224

# Number of image channels (for example 3 in case of RGB, or 1 for grayscale images)
INPUT_CHANNELS = 3
# Number of output masks (1 in case you predict only one type of objects)
OUTPUT_MASK_CHANNELS = 1

if K.image_data_format() == 'channels_first':
    input_shape = (3, IMG_WIDTH, IMG_HEIGHT)
else:
    input_shape = (IMG_WIDTH, IMG_HEIGHT, 3)


def get_train_imgs():
    return ['../data/training/images/train_img_'+str(x)+'.jpg' for x in range(1, 1501)]


def get_train_masks():
    return ['../data/training/masks/train_mask_'+str(x)+'.ppm' for x in range(1, 1501)]


def get_val_imgs():
    return ['../data/validation/images/validation_img_'+str(x)+'.jpg' for x in range(1, 501)]


def get_val_masks():
    return ['../data/validation/masks/validation_mask_'+str(x)+'.ppm' for x in range(1, 501)]


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
    hrz_trans = randint(-20, 20)
    vrt_trans = randint(-20, 20)

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


def generate_data(x_path, y_path, batch):
    mod, idx = len(x_path), 0
    img_shape = (batch, input_shape[0], input_shape[1], input_shape[2])
    mask_shape = (batch, input_shape[0], input_shape[1], 1)

    while True:
        samples = random.sample(range(0, mod), mod)
        samples = [samples[batch*i: batch*i+batch] for i in range(mod//batch)]

        for smpl in samples:
            x = np.zeros(shape=img_shape, dtype=np.float32)
            y = np.zeros(shape=mask_shape, dtype=np.float32)

            for i, s in enumerate(smpl):
                img, msk = ndimage.imread(x_path[s]), ndimage.imread(y_path[s])
                msk = msk[:, :, 1:2]
                img, msk = transform_img(img, msk)
                x[i], y[i] = img, msk

            yield (x, y)


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def resize_images_bilinear(X, height_factor=1, width_factor=1, target_height=None, target_width=None, data_format='default'):
    if data_format == 'default':
        data_format = K.image_data_format()
    if data_format == 'channels_first':
        original_shape = K.int_shape(X)
        if target_height and target_width:
            new_shape = tf.constant(np.array((target_height, target_width)).astype('int32'))
        else:
            new_shape = tf.shape(X)[2:]
            new_shape *= tf.constant(np.array([height_factor, width_factor]).astype('int32'))
        X = permute_dimensions(X, [0, 2, 3, 1])
        X = tf.image.resize_bilinear(X, new_shape)
        X = permute_dimensions(X, [0, 3, 1, 2])
        if target_height and target_width:
            X.set_shape((None, None, target_height, target_width))
        else:
            X.set_shape((None, None, original_shape[2] * height_factor, original_shape[3] * width_factor))
        return X
    elif data_format == 'channels_last':
        original_shape = K.int_shape(X)
        if target_height and target_width:
            new_shape = tf.constant(np.array((target_height, target_width)).astype('int32'))
        else:
            new_shape = tf.shape(X)[1:3]
            new_shape *= tf.constant(np.array([height_factor, width_factor]).astype('int32'))
        X = tf.image.resize_bilinear(X, new_shape)
        if target_height and target_width:
            X.set_shape((None, target_height, target_width, None))
        else:
            X.set_shape((None, original_shape[1] * height_factor, original_shape[2] * width_factor, None))
        return X
    else:
        raise Exception('Invalid data_format: ' + data_format)


class BilinearUpSampling2D(Layer):
    def __init__(self, size=(1, 1), target_size=None, data_format='default', **kwargs):
        if data_format == 'default':
            data_format = K.image_data_format()
        self.size = tuple(size)
        if target_size is not None:
            self.target_size = tuple(target_size)
        else:
            self.target_size = None
        assert data_format in {'channels_last', 'channels_first'}, 'data_format must be in {tf, th}'
        self.data_format = data_format
        self.input_spec = [InputSpec(ndim=4)]
        super(BilinearUpSampling2D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            width = int(self.size[0] * input_shape[2] if input_shape[2] is not None else None)
            height = int(self.size[1] * input_shape[3] if input_shape[3] is not None else None)
            if self.target_size is not None:
                width = self.target_size[0]
                height = self.target_size[1]
            return (input_shape[0],
                    input_shape[1],
                    width,
                    height)
        elif self.data_format == 'channels_last':
            width = int(self.size[0] * input_shape[1] if input_shape[1] is not None else None)
            height = int(self.size[1] * input_shape[2] if input_shape[2] is not None else None)
            if self.target_size is not None:
                width = self.target_size[0]
                height = self.target_size[1]
            return (input_shape[0],
                    width,
                    height,
                    input_shape[3])
        else:
            raise Exception('Invalid data_format: ' + self.data_format)

    def call(self, x, mask=None):
        if self.target_size is not None:
            return resize_images_bilinear(x, target_height=self.target_size[0], target_width=self.target_size[1], data_format=self.data_format)
        else:
            return resize_images_bilinear(x, height_factor=self.size[0], width_factor=self.size[1], data_format=self.data_format)

    def get_config(self):
        config = {'size': self.size, 'target_size': self.target_size}
        base_config = super(BilinearUpSampling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def identity_block(kernel_size, filters, stage, block, weight_decay=0., batch_momentum=0.99):

    def f(input_tensor):
        nb_filter1, nb_filter2, nb_filter3 = filters
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a', kernel_regularizer=l2(weight_decay))(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a', momentum=batch_momentum)(x)
        x = Activation('relu')(x)

        x = Conv2D(nb_filter2, (kernel_size, kernel_size),
                          padding='same', name=conv_name_base + '2b', kernel_regularizer=l2(weight_decay))(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b', momentum=batch_momentum)(x)
        x = Activation('relu')(x)

        x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', kernel_regularizer=l2(weight_decay))(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c', momentum=batch_momentum)(x)

        x = Add()([x, input_tensor])
        x = Activation('relu')(x)
        return x
    return f


def conv_block(kernel_size, filters, stage, block, weight_decay=0., strides=(2, 2), batch_momentum=0.99):

    def f(input_tensor):
        nb_filter1, nb_filter2, nb_filter3 = filters
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = Conv2D(nb_filter1, (1, 1), strides=strides,
                          name=conv_name_base + '2a', kernel_regularizer=l2(weight_decay))(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a', momentum=batch_momentum)(x)
        x = Activation('relu')(x)

        x = Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                          name=conv_name_base + '2b', kernel_regularizer=l2(weight_decay))(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b', momentum=batch_momentum)(x)
        x = Activation('relu')(x)

        x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', kernel_regularizer=l2(weight_decay))(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c', momentum=batch_momentum)(x)

        shortcut = Conv2D(nb_filter3, (1, 1), strides=strides,
                                 name=conv_name_base + '1', kernel_regularizer=l2(weight_decay))(input_tensor)
        shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1', momentum=batch_momentum)(shortcut)

        x = Add()([x, shortcut])
        x = Activation('relu')(x)
        return x
    return f


# Atrous-Convolution version of residual blocks
def atrous_identity_block(kernel_size, filters, stage, block, weight_decay=0., atrous_rate=(2, 2), batch_momentum=0.99):

    def f(input_tensor):
        nb_filter1, nb_filter2, nb_filter3 = filters
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a', kernel_regularizer=l2(weight_decay))(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a', momentum=batch_momentum)(x)
        x = Activation('relu')(x)

        x = Conv2D(nb_filter2, (kernel_size, kernel_size), dilation_rate=atrous_rate,
                          padding='same', name=conv_name_base + '2b', kernel_regularizer=l2(weight_decay))(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b', momentum=batch_momentum)(x)
        x = Activation('relu')(x)

        x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', kernel_regularizer=l2(weight_decay))(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c', momentum=batch_momentum)(x)

        x = Add()([x, input_tensor])
        x = Activation('relu')(x)
        return x
    return f


def atrous_conv_block(kernel_size, filters, stage, block, weight_decay=0., strides=(1, 1), atrous_rate=(2, 2), batch_momentum=0.99):

    def f(input_tensor):
        nb_filter1, nb_filter2, nb_filter3 = filters
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = Conv2D(nb_filter1, (1, 1), strides=strides,
                          name=conv_name_base + '2a', kernel_regularizer=l2(weight_decay))(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a', momentum=batch_momentum)(x)
        x = Activation('relu')(x)

        x = Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same', dilation_rate=atrous_rate,
                          name=conv_name_base + '2b', kernel_regularizer=l2(weight_decay))(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b', momentum=batch_momentum)(x)
        x = Activation('relu')(x)

        x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', kernel_regularizer=l2(weight_decay))(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c', momentum=batch_momentum)(x)

        shortcut = Conv2D(nb_filter3, (1, 1), strides=strides,
                                 name=conv_name_base + '1', kernel_regularizer=l2(weight_decay))(input_tensor)
        shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1', momentum=batch_momentum)(shortcut)

        x = Add()([x, shortcut])
        x = Activation('relu')(x)
        return x
    return f


def AtrousFCN_Resnet50_16s(input_shape = None, weight_decay=0., batch_momentum=0.9, batch_shape=None, classes=1):

    if batch_shape:
        img_input = Input(batch_shape=batch_shape)
        image_size = batch_shape[1:3]
    else:
        img_input = Input(shape=input_shape)
        image_size = input_shape[0:2]

    bn_axis = 3

    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1', kernel_regularizer=l2(weight_decay))(img_input)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1', momentum=batch_momentum)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(3, [64, 64, 256], stage=2, block='a', weight_decay=weight_decay, strides=(1, 1), batch_momentum=batch_momentum)(x)
    x = identity_block(3, [64, 64, 256], stage=2, block='b', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [64, 64, 256], stage=2, block='c', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)

    x = conv_block(3, [128, 128, 512], stage=3, block='a', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [128, 128, 512], stage=3, block='b', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [128, 128, 512], stage=3, block='c', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [128, 128, 512], stage=3, block='d', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)

    x = conv_block(3, [256, 256, 1024], stage=4, block='a', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='b', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='c', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='d', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='e', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='f', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)

    x = atrous_conv_block(3, [512, 512, 2048], stage=5, block='a', weight_decay=weight_decay, atrous_rate=(2, 2), batch_momentum=batch_momentum)(x)
    x = atrous_identity_block(3, [512, 512, 2048], stage=5, block='b', weight_decay=weight_decay, atrous_rate=(2, 2), batch_momentum=batch_momentum)(x)
    x = atrous_identity_block(3, [512, 512, 2048], stage=5, block='c', weight_decay=weight_decay, atrous_rate=(2, 2), batch_momentum=batch_momentum)(x)
    #classifying layer
    #x = Conv2D(classes, (3, 3), dilation_rate=(2, 2), kernel_initializer='normal', activation='linear', padding='same', strides=(1, 1), kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(classes, (1, 1), kernel_initializer='he_normal', activation='sigmoid', padding='same', strides=(1, 1), kernel_regularizer=l2(weight_decay))(x)
    x = BilinearUpSampling2D(target_size=tuple(image_size))(x)

    model = Model(img_input, x)
    model.load_weights('pretrain/fcn_resnet50_weights_tf_dim_ordering_tf_kernels.h5', by_name=True)
    return model


def train_res_fcn(model, epochs):
    train_imgs, train_masks, val_imgs, val_masks = get_img_mask()

    steps_per_epoch = int(ceil(len(train_imgs) / BATCH_SIZE))
    validation_steps = int(ceil(len(val_imgs) / BATCH_SIZE))

    attrib = 'fcn'
    model_type = 'res50'
    filename = current_time()+'-'+attrib
    os.mkdir('models/'+filename)

    with open('models/'+filename+'/attrib.txt', 'wt') as file_writer:
        file_writer.write('used '+model_type+'.'+attrib+'.py\n')

    callback_stopping = EarlyStopping(patience=20, monitor='val_loss')
    callback_checkpoint = ModelCheckpoint('models/' + filename + '/model.'+attrib+'.'+model_type+'.hdf5',
                                          monitor='val_loss', save_best_only=True)
    callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                           factor=0.1, patience=4,
                                           verbose=1, min_lr=1e-6)
    callback_csv = CSVLogger('models/' + filename + '/run_log.csv', append=True)
    callback_terminate_nan = TerminateOnNaN()
    model.compile(optimizer=SGD(lr=0.01, momentum=0.9), loss=dice_coef_loss, metrics=[dice_coef])

    model.fit_generator(generate_data(train_imgs, train_masks, BATCH_SIZE),
                        steps_per_epoch=steps_per_epoch,
                        epochs=epochs,
                        verbose=1,
                        callbacks=[callback_stopping,
                                   callback_checkpoint,
                                   callback_reduce_lr,
                                   callback_csv,
                                   callback_terminate_nan],
                        validation_data=generate_data(val_imgs, val_masks, BATCH_SIZE),
                        validation_steps=validation_steps)

    with open('models/'+filename+'/job_completed.txt', 'wt') as file_writer:
        file_writer.write('job finished gracefully\n')
    model.save('models/' + filename + '/model.'+attrib+'.'+model_type+'.final.hdf5')

    return model


def run():
    model = AtrousFCN_Resnet50_16s(input_shape=input_shape, classes=1)
    train_res_fcn(model, 100)


if __name__ == '__main__':
    run()


# end of file
