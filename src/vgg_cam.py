import keras.backend as K
from keras.models import Sequential
from keras.layers.core import Flatten, Dense
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D
from keras.layers.convolutional import ZeroPadding2D


def vggcam(nb_classes, input_shape=(3, None, None), num_input_channels=1024, dim_ordering = "channels_first"):
# Tensorflow Back-end
# def vggcam(nb_classes, input_shape=(None, None, 3), num_input_channels=1024):
    '''
    :param nb_classes: # classes (IMAGENET = 1000)
    :param input_shape: image shape
    :param num_input_channels: channels CAM layer
    :param bounding_box:  Query processing (Oxford/Paris)
    :return: instance of the model VGG-16 CAM
    '''

    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=input_shape, dim_ordering=dim_ordering))
    model.add(Convolution2D(64, 3, 3, activation='relu', dim_ordering=dim_ordering))
    model.add(ZeroPadding2D((1, 1), dim_ordering=dim_ordering))
    model.add(Convolution2D(64, 3, 3, activation='relu', dim_ordering=dim_ordering))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), dim_ordering=dim_ordering))

    model.add(ZeroPadding2D((1, 1), dim_ordering=dim_ordering))
    model.add(Convolution2D(128, 3, 3, activation='relu', dim_ordering=dim_ordering))
    model.add(ZeroPadding2D((1, 1), dim_ordering=dim_ordering))
    model.add(Convolution2D(128, 3, 3, activation='relu', dim_ordering=dim_ordering))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), dim_ordering=dim_ordering))

    model.add(ZeroPadding2D((1, 1), dim_ordering=dim_ordering))
    model.add(Convolution2D(256, 3, 3, activation='relu', dim_ordering=dim_ordering))
    model.add(ZeroPadding2D((1, 1), dim_ordering=dim_ordering))
    model.add(Convolution2D(256, 3, 3, activation='relu', dim_ordering=dim_ordering))
    model.add(ZeroPadding2D((1, 1), dim_ordering=dim_ordering))
    model.add(Convolution2D(256, 3, 3, activation='relu', dim_ordering=dim_ordering))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), dim_ordering=dim_ordering))

    model.add(ZeroPadding2D((1, 1), dim_ordering=dim_ordering))
    model.add(Convolution2D(512, 3, 3, activation='relu', dim_ordering=dim_ordering))
    model.add(ZeroPadding2D((1, 1), dim_ordering=dim_ordering))
    model.add(Convolution2D(512, 3, 3, activation='relu', dim_ordering=dim_ordering))
    model.add(ZeroPadding2D((1, 1), dim_ordering=dim_ordering))
    model.add(Convolution2D(512, 3, 3, activation='relu', dim_ordering=dim_ordering))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), dim_ordering=dim_ordering))

    model.add(ZeroPadding2D((1, 1), dim_ordering=dim_ordering))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='relu5_1', dim_ordering=dim_ordering))
    model.add(ZeroPadding2D((1, 1), dim_ordering=dim_ordering))
    model.add(Convolution2D(512, 3, 3, activation='relu', dim_ordering=dim_ordering))
    model.add(ZeroPadding2D((1, 1), dim_ordering=dim_ordering))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='relu5_3', dim_ordering=dim_ordering))

    # Add another conv layer with ReLU + GAP
    model.add(Convolution2D(num_input_channels, 3, 3, activation='relu', border_mode="same", name='CAM_relu', dim_ordering=dim_ordering))

    # Global Average Pooling
    model.add(GlobalAveragePooling2D(name='CAM_pool', dim_ordering=dim_ordering))

    # Add the W layer
    model.add(Dense(nb_classes, activation='softmax'))

    model.name = "vgg_cam"

    return model

