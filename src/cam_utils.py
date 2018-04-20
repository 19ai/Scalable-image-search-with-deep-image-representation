import keras.backend as K
import numpy as np
import math
from vgg_cam import vggcam

classes_imagenet = 1000


def get_output_layer(model, layer_name):
    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer = layer_dict[layer_name]
    return layer


def extract_feat_cam(model, layer, batch_size, images, top_nclass):
    '''
    :param model: Network  (vgg_cam)
    :param layer: Layer to extract features   (relu5_1)
    :param batch_size: Batch size  (6)
    :param images: data [n_samples,3,H,W]  (500,3,1024,720) or (500,3,720,1024)
    :param top_nclass: number of CAMs to extract (Top predicted N classes)  (1 or 64)
    :param specify_class: (If we know the classes) --> Class Array
    :param roi: Region of Interest given list of classes
    :return: features, cams, class_list , roi
    '''

    num_samples = images.shape[0]  #[500]

    class_list = np.zeros((num_samples, top_nclass), dtype=np.int32)  #[500,1]

    num_it = int(math.floor(num_samples / batch_size))  # [83]
    last_batch = num_samples % batch_size  # [500-83*6=2]
    batch_size_loop = batch_size  # [6]

    # Set convolutional layer to extract the CAMs (CAM_relu layer)
    final_conv_layer = get_output_layer(model, "CAM_relu")  # layer CAM_relu  output.shape [6,1024,64,45]

    # Set layer to extract the features
    conv_layer_features = get_output_layer(model, layer)  # layer relu5_1  output.shape [6,512,64,45]
    f_shape = conv_layer_features.output_shape  #[N,512,14,14]

    features_conv = np.zeros((num_samples, f_shape[1], images.shape[2] // 16, images.shape[3] // 16))  # [500,512,64,45]
    # features_conv = np.zeros((num_samples, 512, images.shape[2] // 16, images.shape[3] // 16))  # [500,512,64,45]
    cams = np.zeros((images.shape[0], top_nclass, images.shape[2] // 16, images.shape[3] // 16), dtype=np.float32)  # [500,1,64,45]
    all_scores = np.zeros((num_samples, classes_imagenet))  # [500,1000]

    # Function to get scores, conv_maps --> Could be implemented outside, bottleneck (fast version have it outside)
    get_output = K.function([model.layers[0].input, K.learning_phase()],
                            [final_conv_layer.output, model.layers[-1].output, conv_layer_features.output])

    # Extract weights from Dense
    weights_fc = model.layers[-1].get_weights()[0]

    for i in range(0, num_it+1):  # range(0,84)
        if i == num_it:
            if last_batch != 0:
                x = images[i*batch_size:batch_size*i+last_batch, :, :, :]
                batch_size_loop = last_batch
            else:
                break
        else:
            x = images[i*batch_size:batch_size*(i+1), :, :, :]

        [conv_outputs, scores, features] = get_output([x, 0])  # FEED DATA

        features_conv[i*batch_size:i*batch_size+features.shape[0], :, :, :] = features

        for ii in range(0, batch_size_loop):  # range(0,6) or range(0,2) for the last minibatch
            indexed_scores = scores[ii].argsort()[::-1]
            for k in range(0, top_nclass):
                w_class = weights_fc[:, indexed_scores[k]]  # w_class 1D array [1024,]
                all_scores[i * batch_size + ii, k] = scores[ii, indexed_scores[k]] # all_score [500,1000]
                cam = np.zeros(dtype=np.float32, shape=conv_outputs.shape[2:4])  # [64,45]
                for ind, w in enumerate(w_class):
                    cam += w * conv_outputs[ii, ind, :, :]
                cam /= np.max(cam)
                cam[np.where(cam < 0)] = 0

                cams[i*batch_size+ii, k, :, :] = cam

                class_list[i*batch_size+ii, k] = indexed_scores[k]

        return features_conv, cams, class_list


def extract_feat_cam_fast(model, get_output_function, layer_feat, batch_size, images, top_nclass, specify_class=None, roi=False):
    '''
    :param model: network
    :param get_output_function: function to extract features
    :param layer_feat: layer features
    :param batch_size: batch size
    :param images: images [num_images, 3, h, w]
    :param top_nclass: top_predicted CAMs
    :param specify_class: Give class array
    :param roi: Region of interest (True / False)
    :return:
    '''

    # width, height of conv5_1 layer
    # 14x14 for 224x224 input image
    # H/16 x W/16 for H x W input image with VGG-16

    num_samples = images.shape[0]
    class_list = np.zeros((num_samples, top_nclass), dtype=np.int32)

    num_it = int(math.floor(num_samples / batch_size))
    last_batch = num_samples % batch_size
    batch_size_loop = batch_size

    f_shape = layer_feat.output_shape

    # Initialize Arrays
    features_conv = np.zeros((num_samples, f_shape[1], images.shape[2] // 16, images.shape[3] // 16))
    cams = np.zeros((images.shape[0], top_nclass, images.shape[2] // 16, images.shape[3] // 16), dtype=np.float32)
    all_scores = np.zeros((num_samples, classes_imagenet))

    # Extract weights from Dense
    weights_fc = model.layers[-1].get_weights()[0]

    for i in range(0, num_it+1):
        if i == num_it:
            if last_batch != 0:
                x = images[i*batch_size:batch_size*i+last_batch, :, :, :]
                batch_size_loop = last_batch
            else:
                break
        else:
            x = images[i*batch_size:batch_size*(i+1), :, :, :]

        [conv_outputs, scores, features] = get_output_function([x, 0])
        features_conv[i*batch_size:i*batch_size+features.shape[0], :, :, :] = features

        for ii in range(0, batch_size_loop):
            indexed_scores = scores[ii].argsort()[::-1]
            for k in range(0, top_nclass):
                w_class = weights_fc[:, indexed_scores[k]]
                all_scores[i * batch_size + ii, k] = scores[ii, indexed_scores[k]]
                cam = np.zeros(dtype=np.float32, shape=conv_outputs.shape[2:4])
                for ind, w in enumerate(w_class):
                    cam += w * conv_outputs[ii, ind, :, :]
                cam /= np.max(cam)
                cam[np.where(cam < 0)] = 0
                cams[i*batch_size+ii, k, :, :] = cam
                class_list[i*batch_size+ii, k] = indexed_scores[k]

    return features_conv, cams, class_list

def test():
    model = vggcam(1000)
    final_conv_layer = get_output_layer(model, "CAM_relu")
    print final_conv_layer

if __name__ == '__main__':
    test()