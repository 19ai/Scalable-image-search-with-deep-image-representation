import numpy as np
import os
import math
import h5py
import sys
from sklearn.neighbors import NearestNeighbors
import sklearn
from scipy.misc import imread, imresize, imsave
import time
import utils as utils
import shutil
import re
import cv2


# Save Ranking for one query
def save_ranking_one_query(data, query_desc, image_names, path, image_name, src_path, dest_path,top_image = 3):
    # data: descriptor_aggregation over 100060 images  [100060,512]
    # query_desc: descriptor_aggregation over the query image  [1,512]
    # image_names: list of name of all images
    # path: path to save txt file for each query image showing ranking for each image from the dataset of 100700 images
    # image_name: name of query image
    for i in range(0, image_names.shape[0]):
        if image_names[i].replace('\n', '') == image_name:
            utils.create_folders(dest_path + image_name + '/')
            shutil.copy(src_path + image_name + '.jpg', dest_path + image_name + '/' + 'A_query_image.jpg')
            data_aux = data[i].copy()
            data[i] = query_desc
            data_local = data
            distances, indices = compute_distances_optim(query_desc, data)
            sys.stdout.flush()
            file = open(path + image_names[i].replace('\n', '') + '.txt', 'w')
            for ind in indices:
                file.write(image_names[ind])
            file.close()
            data[i] = data_aux
            for j, ind in enumerate(indices):
                if j > top_image-1:
                    break
                try:
                    shutil.copy(src_path + image_names[ind].replace('\n', '') + '.jpg',
                                dest_path + image_name + '/' + 'B_relevant_'+ str(j+1) + '.jpg')
                except:
                    pass
            return indices, data_local

def post_processing(query_image, top_images, top_pp_data, num_return = 3):
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    sift = cv2.ORB_create()

    queryImage = cv2.imread(query_image, 0)  # queryImage
    queryKeypoint, queryDes = sift.detectAndCompute(queryImage, None)
    queryDes = np.asarray(queryDes, np.float32)

    list_images = []
    list_counts = []

    for image, pp_data in zip(top_images,top_pp_data):
        try:
            if (len(queryDes) >= 2 and len(pp_data) >= 2):
                local_matches = flann.knnMatch(queryDes, pp_data, k=2)
                local_count = sum(1 for (m, n) in local_matches if m.distance < 0.7 * n.distance)
            else:
                local_count = 0
        except:
            local_count = 0
        list_counts.append(local_count)
        list_images.append(image)

    result_list = [[x, y] for y, x in sorted(zip(list_counts, list_images), reverse=True)]
    images, counts = zip(*result_list[:num_return])
    return images, counts, len(queryKeypoint)

def save_ranking_one_query_from_folder(data, pp_data, query_desc, image_names, raw_image_name, image_name, dest_path):
    compiler = re.compile('\/\d{1,6}\.')
    distances, indices = compute_distances_optim(query_desc, data)
    sys.stdout.flush()
    for i in range(3):
        shutil.copy(image_names[indices[i]].replace('\n', ''),
                    dest_path + raw_image_name + '_{}_predicted.jpg'.format(i))
    top_images = [image_names[indices[i]].replace('\n', '') for i in range(100)]
    top_pp_data = [pp_data[indices[i]] for i in range(100)]
    post_pp_result_images, post_pp_result_score, post_pp_result_query_keypoint = post_processing(image_name, top_images, top_pp_data)
    for i in range(3):
        shutil.copy(post_pp_result_images[i].replace('\n', ''),
                    dest_path + raw_image_name + '_{}_pp_predicted.jpg'.format(i))
    return indices, data, distances[indices[:3]], re.search(compiler, post_pp_result_images[0].replace('\n', '')).group(0)[1:-1], post_pp_result_score, post_pp_result_query_keypoint


def save_ranking_one_query_from_webcam(data, pp_data, query_desc, image_names, raw_image_name, image_name, dest_path):
    compiler = re.compile('\/\d{1,6}\.')
    distances, indices = compute_distances_optim(query_desc, data)
    # print distances[indices[:10]]
    sys.stdout.flush()
    top_images = [image_names[indices[i]].replace('\n', '') for i in range(100)]
    top_pp_data = [pp_data[indices[i]] for i in range(100)]
    post_pp_result_images, post_pp_result_score, post_pp_result_query_keypoint = post_processing(image_name, top_images, top_pp_data)
    for i in range(1):
        shutil.copy(post_pp_result_images[i].replace('\n', ''),
                    dest_path + raw_image_name + '.jpg')
    return indices, data, distances[indices[:3]], re.search(compiler, post_pp_result_images[0].replace('\n', '')).group(0)[1:-1], post_pp_result_score, post_pp_result_query_keypoint

def save_ranking_one_query_from_image(data, query_desc, image_names, image_name, src_path):
    compiler = re.compile('.*\/')
    dest_path = re.search(compiler, src_path).group(0)
    distances, indices = compute_distances_optim(query_desc, data)
    # print distances[indices[:10]]
    sys.stdout.flush()
    shutil.copy(image_names[indices[0]].replace('\n', ''),
                dest_path + image_name + '_predicted.jpg')
    return indices, data

# Compute distances and get list of indices (dot product --> Faster)
def compute_distances_optim(desc, data):
    dist = np.dot(desc, np.transpose(data))
    ind = dist[0].argsort()[::-1]
    return dist[0], ind

