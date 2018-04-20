import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cuda,floatX=float32"
import sys
import getopt
import src.evaluation as eval
from src.vgg_cam import vggcam
from src.cam_utils import extract_feat_cam_fast, get_output_layer
from src.utils import preprocess_images, load_data
from src.pooling_functions import descriptor_aggregation, weighted_cam_pooling
from scipy.misc import imread
import cPickle as pickle
import re
import glob
from keras.callbacks import *
import keras.backend as K

try:
    opts, args = getopt.getopt(sys.argv[1:], "i:n:d:")
    flag_i = False
    flag_n = False
    flag_d = False
except getopt.GetoptError:
    print 'test.py -i <image> -n <normal_directory> -d <deep_directory>'
    sys.exit(2)
for opt, arg in opts:
    if opt == '-i':
        test_path = arg
        flag_i = True
    elif opt == "-n":  # python test.py -n ../datasets/images/test_images/
        test_path = arg
        flag_n = True
    elif opt == "-d":  # python test.py -d ../datasets/images/transformed_images/
        test_path = arg
        flag_d = True

if not flag_i and not flag_n and not flag_d:
    test_path = 'datasets/images/transformed_images/'
    flag_d = True

maps = list()
t_vector = list()
count = 0
num_cams = 64
top_n_ranking = 100
n_expand = 10
mean_value = [123.68, 116.779, 103.939]  # Need to re-calculate for painting dataset
nb_classes = 1000
size_v = [720, 1024]
size_h = [1024, 720]
VGGCAM_weight_path = 'datasets/pretrained_model/vgg_cam_weights.h5'
model_name = 'vgg_16_CAM'
layer = 'relu5_1'
num_classes_pca = 1
pca_path = 'output/pca/'
train_list_path_h = "../datasets/lists/list_painting_horizontal.txt"
train_list_path_v = "../datasets/lists/list_painting_vertical.txt"
image_names = list()
with open(train_list_path_h, "r") as f:
    for line in f:
        image_names.append(line)
with open(train_list_path_v, "r") as f:
    for line in f:
        image_names.append(line)

image_names = np.array(image_names)
path_aggregated_descriptors = 'output/aggregated_descriptors/'
aggregated_descriptors_path_wp = path_aggregated_descriptors + 'painting_aggregated_descriptors_wp.h5'

model = vggcam(nb_classes)
model.load_weights(VGGCAM_weight_path)
final_conv_layer = get_output_layer(model, "CAM_relu")
conv_layer_features = get_output_layer(model, layer)
get_output = K.function([model.layers[0].input, K.learning_phase()],
                        [final_conv_layer.output, model.layers[-1].output, conv_layer_features.output])
data = load_data(aggregated_descriptors_path_wp)

with open(pca_path + 'pca_matrix_from', 'rb') as file:
    pca_matrix = pickle.load(file)
print "Load completed"
compiler = re.compile('\/[^\/]+\.jpg')

if flag_d:
    for subdir, dirs, files in os.walk(test_path):
        for file in files:
            image_name = subdir + os.sep + file
            if image_name.endswith("TransformedImage.jpg"):
                t = time.time()
                img = imread(image_name, mode='RGB')
                raw_image_name = re.search(compiler, image_name).group(0)[1:-4]
                if img.shape[0] > img.shape[1]:
                    size = size_v
                else:
                    size = size_h
                img_p = preprocess_images(img, size[0], size[1], mean_value)
                features, cams, class_list = extract_feat_cam_fast(model, get_output, conv_layer_features, 1, img_p,
                                                                   num_cams)
                d_wp = weighted_cam_pooling(features, cams)
                desc = descriptor_aggregation(d_wp, 1, num_cams, pca_matrix)
                indices_local, data_local = eval.save_ranking_one_query_from_folder(data, desc, image_names,
                                                                                    raw_image_name,
                                                                                    subdir + os.sep)

                print 'Time for one query image: ', time.time() - t

elif flag_n:
    for image_name in glob.iglob(test_path + '*.jpg'):
        t = time.time()
        img = imread(image_name, mode='RGB')
        raw_image_name = re.search(compiler, image_name).group(0)[1:-4]
        if img.shape[0] > img.shape[1]:
            size = size_v
        else:
            size = size_h
        img_p = preprocess_images(img, size[0], size[1], mean_value)
        features, cams, class_list = extract_feat_cam_fast(model, get_output, conv_layer_features, 1, img_p,
                                                           num_cams)
        d_wp = weighted_cam_pooling(features, cams)
        desc = descriptor_aggregation(d_wp, 1, num_cams, pca_matrix)
        indices_local, data_local = eval.save_ranking_one_query_from_folder(data, desc, image_names, raw_image_name,
                                                                            test_path)

        print 'Time for one query image: ', time.time() - t

elif flag_i:
    image_name = test_path
    img = imread(image_name, mode='RGB')
    raw_image_name = re.search(compiler, image_name).group(0)[1:-4]
    if img.shape[0] > img.shape[1]:
        size = size_v
    else:
        size = size_h
    img_p = preprocess_images(img, size[0], size[1], mean_value)
    features, cams, class_list = extract_feat_cam_fast(model, get_output, conv_layer_features, 1, img_p,
                                                       num_cams)
    d_wp = weighted_cam_pooling(features, cams)
    desc = descriptor_aggregation(d_wp, 1, num_cams, pca_matrix)
    indices_local, data_local = eval.save_ranking_one_query_from_image(data, desc, image_names, raw_image_name,
                                                                       test_path)
