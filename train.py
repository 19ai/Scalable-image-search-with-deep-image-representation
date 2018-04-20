import os

os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cuda0,floatX=float32"
from scipy.misc import imread
import sys
from src.cam_utils import extract_feat_cam
from src.vgg_cam import vggcam
from src.utils import create_folders, save_data, preprocess_images,load_data
from src.pooling_functions import weighted_cam_pooling, descriptor_aggregation, retrieve_n_descriptors, compute_pca
import pickle
from keras.callbacks import *

LAYER = 'relu5_1'
IMAGE_BATCH_SIZE = 1000
DESCRIPTORS_BATCH_SIZE = 10000
DIM_DESCRIPTOR = 512
train_list_path_h = "datasets/lists/list_painting_horizontal.txt"
train_list_path_v = "datasets/lists/list_painting_vertical.txt"
path_descriptors = 'output/cam_descriptors/'
descriptors_cams_path_wp = path_descriptors + 'painting_cam_descriptors_wp_'
create_folders(path_descriptors)
batch_size_re = 6
n_images_painting = 43690
num_classes_pca = 1
pca_dim = 512
num_cams = 64
num_prec_classes = 64
path_aggregated_descriptors = 'output/aggregated_descriptors/'
aggregated_descriptors_path_wp = path_aggregated_descriptors + 'painting_aggregated_descriptors_wp.h5'
pca_path = 'output/pca/'
create_folders(pca_path)
create_folders(path_aggregated_descriptors)

def extract_cam_descriptors(model, batch_size, num_classes, size, mean_value, image_train_list_path, desc_wp,
                            chunk_index, ind=0):
    images = [0] * IMAGE_BATCH_SIZE
    image_names = [0] * IMAGE_BATCH_SIZE
    counter = 0
    desc_count = 0
    num_images = 0
    for line in open(image_train_list_path):
        if counter >= IMAGE_BATCH_SIZE:
            data = preprocess_images(images, size[0], size[1], mean_value)
            features, cams, cl = \
                extract_feat_cam(model, LAYER, batch_size, data, num_classes)
            d_wp = weighted_cam_pooling(features, cams)
            desc_wp = np.concatenate((desc_wp, d_wp))
            sys.stdout.flush()
            counter = 0
            desc_count += IMAGE_BATCH_SIZE
            if DESCRIPTORS_BATCH_SIZE == desc_count:
                print 'Saving ...' + descriptors_cams_path_wp  + str(chunk_index) + '.h5'
                save_data(desc_wp, descriptors_cams_path_wp  + str(chunk_index) + '.h5', '')
                desc_count = 0
                chunk_index += 1
                desc_wp = np.zeros((0, DIM_DESCRIPTOR), dtype=np.float32)
            ind += 1
        line = line.rstrip('\n')
        images[counter] = imread(line, mode='RGB')
        image_names[counter] = line
        counter += 1
        num_images += 1

    # Last batch
    data = np.zeros((counter, 3, size[1], size[0]), dtype=np.float32)
    data[0:] = preprocess_images(images[0:counter], size[0], size[1], mean_value)
    features, cams, cl = extract_feat_cam(model, LAYER, batch_size, data, num_classes)
    d_wp = weighted_cam_pooling(features, cams)
    desc_wp = np.concatenate((desc_wp, d_wp))
    save_data(desc_wp, descriptors_cams_path_wp + str(chunk_index) + '.h5', '')
    chunk_index += 1
    desc_wp = np.zeros((0, DIM_DESCRIPTOR), dtype=np.float32)
    ind += 1
    sys.stdout.flush()

    return desc_wp, chunk_index


def train():
    t_0 = time.time()
    # Horizontal Images
    size_h = [1024, 720]
    # Vertical Images
    size_v = [720, 1024]
    # Model parameters
    mean_value = [123.68, 116.779, 103.939]
    num_classes = 64
    nb_classes = 1000
    VGGCAM_weight_path = 'datasets/pretrained_model/vgg_cam_weights.h5'
    model = vggcam(nb_classes)
    model.load_weights(VGGCAM_weight_path)
    model.summary()
    batch_size = 6
    chunk_index = 0
    ind = 0
    n_chunks = 4
    desc_wp = np.zeros((0, DIM_DESCRIPTOR), dtype=np.float32)
    # Horizontal Images
    desc_wp, c_ind = \
        extract_cam_descriptors(model, batch_size, num_classes, size_h, mean_value, train_list_path_h, desc_wp,
                                chunk_index)
    # Vertical Images
    desc_wp, c_ind = \
        extract_cam_descriptors(model, batch_size, num_classes, size_v, mean_value, train_list_path_v, desc_wp, c_ind,
                                ind)
    print 'Total time elapsed: ', time.time() - t_0
    pcas = np.zeros((0, 512), dtype=np.float32)
    for n_in in range(0, n_chunks + 1):
        pca = load_data(descriptors_cams_path_wp + str(n_in) + '.h5')
        pcas = np.concatenate((pcas, pca))
    pca_desc = retrieve_n_descriptors(num_classes_pca, n_images_painting, pcas)
    pca_matrix = compute_pca(pca_desc, pca_dim=pca_dim, whiten=True)
    with open(pca_path + 'pca_matrix_from', 'wb') as file:
        pickle.dump(pca_matrix, file)
    data = np.zeros((0, 512), dtype=np.float32)
    for n_in in range(0, n_chunks + 1):
        desc = load_data(descriptors_cams_path_wp + str(n_in) + '.h5')
        data = np.concatenate((data, descriptor_aggregation(desc, desc.shape[0] / num_prec_classes,
                                                            num_cams, pca_matrix)))
        t = time.time()
        print 'Time elapsed loading: ', time.time() - t
        sys.stdout.flush()
    print "descriptor's shape for the dataset: ", data.shape
    save_data(data, aggregated_descriptors_path_wp, '')
    print 'Total time elapsed: ', time.time() - t_0

if __name__ == '__main__':
    train()
