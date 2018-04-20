import os

os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cuda0,floatX=float32"
import sys
import getopt
import src.evaluation as eval
from src.vgg_cam import vggcam
from src.cam_utils import extract_feat_cam_fast, get_output_layer
from src.utils import preprocess_images, load_data, load_pp_data
from src.pooling_functions import descriptor_aggregation, weighted_cam_pooling
from scipy.misc import imread
import numpy as np
import cPickle as pickle
import re
import shutil
import glob
from PIL import Image
from distutils.dir_util import copy_tree
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
    elif opt == "-n":  # python test.py -n datasets/images/test_images/
        test_path = arg
        flag_n = True
    elif opt == "-d":  # python test.py -d ../datasets/images/transformed_images/
        test_path = arg
        flag_d = True

if not flag_i and not flag_n and not flag_d:
    test_path = 'datasets/images/transformed_images/'
    flag_d = True

first_wrong_prediction_path = 'output/statistics/first_wrong_prediction'
pp_wrong_prediction_path = 'output/statistics/pp_wrong_prediction'
if os.path.isdir(first_wrong_prediction_path):
    shutil.rmtree(first_wrong_prediction_path)
os.mkdir(first_wrong_prediction_path)

if os.path.isdir(pp_wrong_prediction_path):
    shutil.rmtree(pp_wrong_prediction_path)
os.mkdir(pp_wrong_prediction_path)

TEXT_FILE_PATTERN = 'Image Cosine_Similarity Matching_Point KP/MP Original_image_resolution Query_image_resolution BPP-APP\n'

general_output_file = open('output/statistics/general_output.txt', 'w')
general_output_file.write(TEXT_FILE_PATTERN)

maps = list()
t_vector = list()
count = 0
first_wrong = 0
pp_wrong = 0
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
# image_path = None
train_list_path_h = "datasets/lists/list_painting_horizontal.txt"
train_list_path_v = "datasets/lists/list_painting_vertical.txt"
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

pp_descriptor_path = 'output/pp_descriptor/painting_pp_descriptors_wp.h5'
pp_data = [load_pp_data(pp_descriptor_path, index) for index in range(num_images)]
with open(pca_path + 'pca_matrix_from', 'rb') as file:
    pca_matrix = pickle.load(file)
print "Load completed"
compiler = re.compile('\/[^\/]+\.jpg')

if flag_d:
    for subdir, dirs, files in os.walk(test_path):
        for file in files:
            image_name = subdir + os.sep + file
            if image_name.endswith("TransformedImage.jpg"):
                count += 1
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
                _, _, distance, _, score, max_score = eval.save_ranking_one_query_from_folder(data, pp_data, desc, image_names,
                                                                                              raw_image_name,
                                                                                              image_name,
                                                                                              subdir + os.sep)
                if max_score != 0:
                    detected_matching_percentage = float(score[0]) / max_score
                else:
                    detected_matching_percentage = 0

                original_image = Image.open(subdir + os.sep + 'OriginalImage.jpg')
                before_pp_image = Image.open(subdir + os.sep + raw_image_name + '_0_predicted.jpg')
                pp_image = Image.open(subdir + os.sep + raw_image_name + '_0_pp_predicted.jpg')
                x_compiler = re.compile('\d{1,6}\_\d{1,3}')

                TEXT_CONTENT_PATTERN = '{} {} {} {}/{} {}x{} {}x{}'.format(re.search(x_compiler, subdir).group(0), distance[0],
                                                                             detected_matching_percentage,
                                                                             score[0],
                                                                             max_score, original_image.size[1],
                                                                             original_image.size[0], img.shape[0],
                                                                             img.shape[1])
                if list(original_image.getdata()) != list(before_pp_image.getdata()):
                    is_first_prediction_correct = False
                    copy_tree(subdir, first_wrong_prediction_path + os.sep + re.search(x_compiler, subdir).group(0))
                    first_wrong += 1
                    print "There are {}/{} wrong first predictions ({} %)".format(first_wrong, count,
                                                                                  float(first_wrong * 100) / count)


                else:
                    is_first_prediction_correct = True

                if list(original_image.getdata()) != list(pp_image.getdata()):
                    copy_tree(subdir, pp_wrong_prediction_path + os.sep + re.search(x_compiler, subdir).group(0))
                    pp_wrong += 1

                    if is_first_prediction_correct:
                        general_output_file.write(TEXT_CONTENT_PATTERN +' correct-incorrect\n')
                    else:
                        general_output_file.write(TEXT_CONTENT_PATTERN +' incorrect-incorrect\n')
                    print "There are {}/{} wrong post processing predictions ({} %)".format(pp_wrong, count, float(
                        pp_wrong * 100) / count)

                else:
                    if is_first_prediction_correct:
                        general_output_file.write(TEXT_CONTENT_PATTERN +' correct-correct\n')
                    else:
                        general_output_file.write(TEXT_CONTENT_PATTERN +' incorrect-correct\n')

                        # print 'Time for one query image: ', time.time() - t
                shutil.rmtree(subdir)
    general_output_file.close()


elif flag_n:
    # ranking_image_path = '../output/final_outputs/test_images/'
    # shutil.rmtree(ranking_image_path, ignore_errors=True)
    # create_folders(ranking_image_path)
    top_3_file = open('output/statistics/top_3_cosine_similarities.txt', 'w')
    top_3_file.write("Image Number_1 Number_2 Number_3\n")
    if test_path[-1] != '/':
        test_path += '/'
    for image_name in glob.iglob(test_path + '*.jpg'):
        if "predicted" in image_name:
            continue
        count += 1
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
        _, _, distance, _, score, max_score = eval.save_ranking_one_query_from_folder(data, pp_data, desc, image_names, raw_image_name,
                                                                            image_name,
                                                                            test_path)
        if max_score != 0:
            detected_matching_percentage = float(score[0]) / max_score
        else:
            detected_matching_percentage = 0

        original_image = Image.open(image_name)
        before_pp_image = Image.open(test_path + raw_image_name + '_0_predicted.jpg')
        pp_image = Image.open(test_path + raw_image_name + '_0_pp_predicted.jpg')
        # x_compiler = re.compile('\d{1,6}\_\d{1,3}')

        TEXT_CONTENT_PATTERN = '{} {} {} {}/{} {}x{} {}x{}'.format(raw_image_name, distance[0],
                                                                   detected_matching_percentage,
                                                                   score[0],
                                                                   max_score, original_image.size[1],
                                                                   original_image.size[0], img.shape[0],
                                                                   img.shape[1])
        if list(original_image.getdata()) != list(before_pp_image.getdata()):
            is_first_prediction_correct = False
            first_wrong += 1
            print "There are {}/{} wrong first predictions ({} %)".format(first_wrong, count,
                                                                          float(first_wrong * 100) / count)


        else:
            is_first_prediction_correct = True

        if list(original_image.getdata()) != list(pp_image.getdata()):
            pp_wrong += 1

            if is_first_prediction_correct:
                general_output_file.write(TEXT_CONTENT_PATTERN + ' correct-incorrect\n')
            else:
                general_output_file.write(TEXT_CONTENT_PATTERN + ' incorrect-incorrect\n')
            print "There are {}/{} wrong post processing predictions ({} %)".format(pp_wrong, count, float(
                pp_wrong * 100) / count)

        else:
            if is_first_prediction_correct:
                general_output_file.write(TEXT_CONTENT_PATTERN + ' correct-correct\n')
            else:
                general_output_file.write(TEXT_CONTENT_PATTERN + ' incorrect-correct\n')

        top_3_file.write("{} {} {} {}".format(raw_image_name, distance[0], distance[1], distance[2]))
        print 'Time for one query image: ', time.time() - t
    general_output_file.close()
    top_3_file.close()


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
