from src.utils import create_folders, save_pp_data, preprocess_images,load_pp_data
import cv2
import numpy as np
import os
train_list_path_h = "datasets/lists/list_painting_horizontal.txt"
train_list_path_v = "datasets/lists/list_painting_vertical.txt"
pp_descriptor_path = 'output/pp_descriptor/painting_pp_descriptors_wp.h5'
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)  # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params, search_params)
sift = cv2.ORB_create()


def post_processing(query_image):
    queryImage = cv2.imread(query_image, 0)  # queryImage
    _, queryDes = sift.detectAndCompute(queryImage, None)
    queryDes = np.asarray(queryDes, np.float32)
    return queryDes

def train():
    index = 0
    for line in open(train_list_path_h):
        des = post_processing(line.replace('\n', ''))
        print index, des.shape
        save_pp_data(des, pp_descriptor_path, '',index)
        index += 1
    for line in open(train_list_path_v):
        des = post_processing(line.replace('\n', ''))
        print index, des.shape
        save_pp_data(des, pp_descriptor_path, '',index)
        index += 1


if __name__ == '__main__':
    if os.path.exists(pp_descriptor_path):
        os.remove(pp_descriptor_path)
    train()
