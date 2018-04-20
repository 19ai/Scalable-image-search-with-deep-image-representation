from scipy.misc import imread, imresize, imsave
import numpy as np
import os

import h5py

def create_folders(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_data(filepath):
    with h5py.File(filepath, 'r') as hf:
        data = np.array(hf.get('data'))
        return data


def save_data(data, path, name):
    with h5py.File(path + name, 'w') as hf:
        hf.create_dataset('data', data=data)

def load_data(filepath):
    with h5py.File(filepath, 'r') as hf:
        data = np.array(hf.get('data'))
        return data


def save_pp_data(data, path, name, index):
    with h5py.File(path + name, 'a') as hf:
        hf.create_dataset(str(index), data=data)

def load_pp_data(filepath, index):
    with h5py.File(filepath, 'r') as hf:
        data = np.array(hf.get(str(index)))
        # print 'Shape of the array features: ', data.shape
        return data

def preprocess_images(images, img_width, img_height, mean_value):
    if isinstance(images, list):
        num_images = len(images)
        # print "Preprocessing " + str(num_images) + ' Images...'
        x = np.zeros((num_images, 3, img_height, img_width), dtype=np.float32)

        for i in range(0, num_images):
            # print images[i].shape
            images[i] = imresize(images[i], [img_height, img_width]).astype(dtype=np.float32)

            # RGB -> BGR
            R = np.copy(images[i][:, :, 0])
            B = np.copy(images[i][:, :, 2])
            images[i][:, :, 0] = B
            images[i][:, :, 2] = R

            # Subtract mean
            images[i][:, :, 0] -= mean_value[0]
            images[i][:, :, 1] -= mean_value[1]
            images[i][:, :, 2] -= mean_value[2]

            x[i, :, :, :] = np.transpose(images[i], (2, 0, 1))
        return x

    else:
        # print 'Preprocessing Image...'
        images = imresize(images, [img_height, img_width]).astype(dtype=np.float32)
        x = np.zeros((1, 3, img_height, img_width), dtype=np.float32)

        # RGB -> BGR
        R = np.copy(images[:, :, 0])
        B = np.copy(images[:, :, 2])
        images[:, :, 0] = B
        images[:, :, 2] = R

        # Subtract mean
        images[:, :, 0] -= mean_value[0]
        images[:, :, 1] -= mean_value[1]
        images[:, :, 2] -= mean_value[2]

        x[0] = np.transpose(images, (2, 0, 1))
        return x


def print_classes(dictionary_labels, vector_classes):
    class_list = list()
    for vc in vector_classes:
        print dictionary_labels[vc]
        class_list.append(dictionary_labels[vc])
    return class_list
