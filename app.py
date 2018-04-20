#!flask/bin/python
import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cuda0,floatX=float32"
import random
import string
import urllib
from PIL import Image
from scipy.misc import imsave
import re
import src.evaluation as eval
from src.vgg_cam import vggcam
from src.cam_utils import extract_feat_cam_fast, get_output_layer
from src.utils import preprocess_images, load_data, load_pp_data, camstream
from src.pooling_functions import descriptor_aggregation, weighted_cam_pooling
from scipy.misc import imread
import cPickle as pickle
import shutil
import pandas as pd
from keras.callbacks import *
import keras.backend as K
from src.crop_image import auto_crop

SERVER = "nguyenv@atcremers17.informatik.tu-muenchen.de:"
PORT = "58022"

from flask import Flask, request, render_template, send_from_directory, flash, session, redirect, url_for
from users.create_user_database import *

ADMIN_NAME = ['viet']
engine = create_engine('sqlite:///painting.db', echo=True)
create_database('users/user_database.csv')
__author__ = 'Viet Nguyen'

app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
DATABASE_FOLDER = 'databases/'
rand_str = lambda n: ''.join([random.choice(string.lowercase + string.digits) for i in xrange(n)])


@app.route('/')
def home():
    global is_load
    if is_load == False:
        global compiler, size_h, size_v, mean_value, model, get_output, conv_layer_features, num_cams, pca_matrix, data, pp_data, image_names, xl
        num_cams = 64
        mean_value = [123.68, 116.779, 103.939]  # Need to re-calculate for painting dataset
        nb_classes = 1000
        size_v = [720, 1024]
        size_h = [1024, 720]
        VGGCAM_weight_path = 'datasets/pretrained_model/vgg_cam_weights.h5'
        layer = 'relu5_1'
        pca_path = 'output/pca/'
        train_list_path_h = "datasets/lists/list_painting_horizontal.txt"
        train_list_path_v = "datasets/lists/list_painting_vertical.txt"
        image_names = list()
        with open(train_list_path_h, "r") as f:
            for line in f:
                image_names.append(line)
        with open(train_list_path_v, "r") as f:
            for line in f:
                image_names.append(line)
        num_images = len(image_names)
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

        pp_descriptor_path = 'output/pp_descriptor/painting_pp_descriptors_wp.h5'
        pp_data = [load_pp_data(pp_descriptor_path, index) for index in range(num_images)]
        compiler = re.compile('\/[^\/]+\.\w{3,4}')
        is_load = True
        xl = pd.read_excel(DATABASE_FOLDER + "catalog.xls", sheetname='catalog').values
        print "Load completed"

    return render_template('login.html')


@app.route('/login', methods=['POST'])
def do_admin_login():
    POST_USERNAME = str(request.form['username'])
    POST_PASSWORD = str(request.form['password'])

    Session = sessionmaker(bind=engine)
    s = Session()
    query = s.query(User).filter(User.username.in_([POST_USERNAME]), User.password.in_([POST_PASSWORD]))
    result = query.first()
    if result:
        session[result.username] = True
        user_folder = os.path.join(APP_ROOT, 'users', result.username)
        if os.path.isdir(user_folder):
            shutil.rmtree(user_folder)
        os.mkdir(user_folder)
        return redirect(url_for('index', user=result.username))
    else:
        flash('Wrong password! Please try again')
        return home()


@app.route("/logout/<user>")
# def logout():
def logout(user):
    session[user] = False
    return home()


@app.route("/<user>")
def index(user="Noone"):
    return render_template("upload.html", user=user)


@app.route("/<user>/upload", methods=["POST"])
def upload(user):
    target = os.path.join(APP_ROOT, 'users', user)
    if os.path.isdir(target):
        shutil.rmtree(target)
    os.mkdir(target)
    compiler = re.compile('\.\w{3,4}')
    for upload in request.files.getlist("file"):
        filename = upload.filename
        new_name = rand_str(20) + re.search(compiler, filename).group(0)
        destination = "/".join([target, new_name])
        print ("Save it to:", destination)
        upload.save(destination)
        im = Image.open(destination)
        im.thumbnail([600, 1000], Image.ANTIALIAS)
        im.save(destination)
    return render_template("crop.html", user=user, path=user, image_name=new_name,
                           width=im.size[0], height=im.size[1], rotation=0)


@app.route("/<user>/upload_1", methods=["POST"])
def upload_1(user):
    target = os.path.join(APP_ROOT, 'users', user)
    if os.path.isdir(target):
        shutil.rmtree(target)
    os.mkdir(target)
    new_name = rand_str(20) + '.jpg'
    destination = "/".join([target, new_name])
    url = request.form['name']
    urllib.urlretrieve(url, destination)
    im = Image.open(destination)
    im.thumbnail([600, 1000], Image.ANTIALIAS)
    im.save(destination)
    return render_template("crop.html", user=user, path=user, image_name=new_name,
                           width=im.size[0], height=im.size[1], rotation=0)


@app.route("/<user>/upload_2", methods=["POST"])
def upload_2(user):
    target = os.path.join(APP_ROOT, 'users', user)
    if os.path.isdir(target):
        shutil.rmtree(target)
    os.mkdir(target)
    new_name = rand_str(20) + '.jpg'
    destination = "/".join([target, new_name])
    camstream(destination)
    print destination
    im = Image.open(destination)
    im.thumbnail([600, 1000], Image.ANTIALIAS)
    im.save(destination)
    return render_template("crop.html", user=user, path=user, image_name=new_name,
                           width=im.size[0], height=im.size[1], rotation=0)


@app.route("/<user>/auto_crop/<filename>/<before_rr>_<new_rr>")
def crop(user, filename, before_rr, new_rr):
    target = os.path.join(APP_ROOT, 'users', user)
    if 'auto_crop_' in filename:
        destination = "/".join([target, filename])
        im = Image.open(destination)
        return render_template("crop.html", user=user, path=user, image_name=filename,
                               width=im.size[0], height=im.size[1], cropped=1, rotation=int(before_rr))
    else:
        cropped_width, cropped_height = auto_crop(target, filename)
        return render_template("crop.html", user=user, path=user, image_name='auto_crop_' + filename,
                               width=cropped_width,
                               height=cropped_height, cropped=1, rotation=int(before_rr))


@app.route("/<user>/reset/<filename>")
def reset(user, filename):
    target = os.path.join(APP_ROOT, 'users', user)
    filename = re.sub('auto_crop_', '', filename)
    filename = re.sub('rotate_\d{1,3}_', '', filename)
    destination = "/".join([target, filename])
    im = Image.open(destination)
    return render_template("crop.html", user=user, path=user, image_name=filename, width=im.size[0],
                           height=im.size[1], rotation=0)


@app.route("/<user>/rotate/<filename>/<before_rr>_<new_rr>")
def rotate(user, filename, before_rr, new_rr):
    target = os.path.join(APP_ROOT, 'users', user)
    destination = "/".join([target, filename])
    im = Image.open(destination)
    im = im.rotate(90 * int(new_rr), expand=True)
    filename = re.sub('rotate_\d{1,3}_', '', filename)
    rotation = (int(before_rr) + int(new_rr)) % 4
    filename = "rotate_{}_".format(90 * rotation) + filename
    destination = "/".join([target, filename])
    im.save(destination)
    return render_template("crop.html", user=user, path=user, image_name=filename, width=im.size[0],
                           height=im.size[1], rotation=rotation)


@app.route('/<user>/display/<filename>', methods=["POST"])
def display(user, filename):
    BEFORE_TT_THRESHOLD_1 = 0.2
    BEFORE_TT_THRESHOLD_2 = 0.91
    AFTER_PP_THRESHOLD_1 = 10
    AFTER_PP_THRESHOLD_2 = 20

    x = int(request.form['x'])
    y = int(request.form['y'])
    w = int(request.form['w'])
    h = int(request.form['h'])

    utc_datetime = datetime.datetime.utcnow()
    current_time = utc_datetime.strftime("%Y-%m-%d %H:%M:%S")
    current_user_directory = os.path.join(admin_directory, '{}_{}'.format(user, current_time))
    os.makedirs(current_user_directory)
    information_dict = {}
    information_dict['Time_stamp'] = current_time

    im = imread(os.path.join(APP_ROOT, 'users', user, filename), mode='RGB')
    shutil.copy(os.path.join(APP_ROOT, 'users', user, filename), current_user_directory + os.sep + 'uploaded_image.jpg')
    original_width = im.shape[1]
    original_height = im.shape[0]

    query_image_path = os.path.join('users', user)
    new_compiler = re.compile('\.\w{3,4}')
    new_file_name = rand_str(20) + re.search(new_compiler, filename).group(0)

    imsave(query_image_path + '/cropped_' + new_file_name, im[y:y + h, x:x + w, :])
    shutil.copy(query_image_path + '/cropped_' + new_file_name, current_user_directory + os.sep + 'query_image.jpg')

    # for image_name in glob.iglob(query_image_path + '/cropped_{}'.format(new_file_name)):
    image_name = query_image_path + '/cropped_' + new_file_name
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
    indices_local, data_local, distances, best_index, scores, _ = eval.save_ranking_one_query_from_folder(data, pp_data,
                                                                                                          desc,
                                                                                                          image_names,
                                                                                                          raw_image_name,
                                                                                                          image_name,
                                                                                                          query_image_path + os.sep)

    print 'Time for one query image: ', time.time() - t

    initial_result_images = []
    # show_image_path = os.path.join(user, 'query_image')
    show_image_path = user

    initial_result_images.append([show_image_path, 'cropped_' + new_file_name])
    initial_result_images.append([show_image_path, 'cropped_' + new_file_name[:-4] + '_0_predicted.jpg'])
    initial_result_images.append([show_image_path, 'cropped_' + new_file_name[:-4] + '_1_predicted.jpg'])
    initial_result_images.append([show_image_path, 'cropped_' + new_file_name[:-4] + '_2_predicted.jpg'])
    shutil.copy(os.path.join(APP_ROOT, 'users', user, 'cropped_' + new_file_name[:-4] + '_0_predicted.jpg'),
                current_user_directory + os.sep + 'before_PP_1.jpg')
    shutil.copy(os.path.join(APP_ROOT, 'users', user, 'cropped_' + new_file_name[:-4] + '_1_predicted.jpg'),
                current_user_directory + os.sep + 'before_PP_2.jpg')
    shutil.copy(os.path.join(APP_ROOT, 'users', user, 'cropped_' + new_file_name[:-4] + '_2_predicted.jpg'),
                current_user_directory + os.sep + 'before_PP_3.jpg')

    pp_images = []
    pp_images.append([show_image_path, 'cropped_' + new_file_name[:-4] + '_0_pp_predicted.jpg'])
    pp_images.append([show_image_path, 'cropped_' + new_file_name[:-4] + '_1_pp_predicted.jpg'])
    pp_images.append([show_image_path, 'cropped_' + new_file_name[:-4] + '_2_pp_predicted.jpg'])
    shutil.copy(os.path.join(APP_ROOT, 'users', user, 'cropped_' + new_file_name[:-4] + '_0_pp_predicted.jpg'),
                current_user_directory + os.sep + 'after_PP_1.jpg')
    shutil.copy(os.path.join(APP_ROOT, 'users', user, 'cropped_' + new_file_name[:-4] + '_1_pp_predicted.jpg'),
                current_user_directory + os.sep + 'after_PP_2.jpg')
    shutil.copy(os.path.join(APP_ROOT, 'users', user, 'cropped_' + new_file_name[:-4] + '_2_pp_predicted.jpg'),
                current_user_directory + os.sep + 'after_PP_3.jpg')
    scores = np.array(scores)

    text_file = open(current_user_directory + os.sep + 'information.txt', 'w')
    text_file.write('User: {}\n'.format(user))
    text_file.write('Query time: {}\n'.format(current_time))
    text_file.write('Uploaded image path: {}\n'.format(current_user_directory + os.sep + 'uploaded_image.jpg'))
    text_file.write('Query image path: {}\n'.format(current_user_directory + os.sep + 'query_image.jpg'))

    if user in ADMIN_NAME:
        if (float(distances[0]) > BEFORE_TT_THRESHOLD_2) or (
                            BEFORE_TT_THRESHOLD_1 < float(distances[0]) <= BEFORE_TT_THRESHOLD_2 and float(
                    scores[0]) >= AFTER_PP_THRESHOLD_2):
            text_file.write('Output (found result): \n')
            text_file.write(
                'Top {} scores (cosine similarity) before post-processing: {}, {}, {} \n'.format(3, distances[0],
                                                                                                 distances[1],
                                                                                                 distances[2]))
            text_file.write(
                'Top {} scores (matching point) after post-processing: {}, {}, {} \n'.format(3, scores[0], scores[1],
                                                                                             scores[2]))
            text_file.close()
            return render_template("result_admin.html", initial_images=initial_result_images, pp_images=pp_images,
                                   width=original_width, height=original_height, distance=distances, score=scores,
                                   information=xl[int(best_index) - 1], user=user)

        elif float(distances[0]) < BEFORE_TT_THRESHOLD_1 or (
                            BEFORE_TT_THRESHOLD_1 < float(distances[0]) <= BEFORE_TT_THRESHOLD_2 and float(
                    scores[0]) < AFTER_PP_THRESHOLD_1):
            error = True
            text_file.write('Output (no result): \n')
            text_file.write(
                'Top {} scores (cosine similarity) before post-processing: {}, {}, {} \n'.format(3, distances[0],
                                                                                                 distances[1],
                                                                                                 distances[2]))
            text_file.write(
                'Top {} scores (matching point) after post-processing: {}, {}, {} \n'.format(3, scores[0], scores[1],
                                                                                             scores[2]))
            text_file.close()
            return render_template("result_admin.html", initial_images=initial_result_images, pp_images=pp_images,
                                   width=original_width, height=original_height, distance=distances, score=scores,
                                   information=xl[int(best_index) - 1], user=user, error1=error)
        else:
            error = True
            text_file.write('Output (uncertain result): \n')
            text_file.write(
                'Top {} scores (cosine similarity) before post-processing: {}, {}, {} \n'.format(3, distances[0],
                                                                                                 distances[1],
                                                                                                 distances[2]))
            text_file.write(
                'Top {} scores (matching point) after post-processing: {}, {}, {} \n'.format(3, scores[0], scores[1],
                                                                                             scores[2]))
            text_file.close()
            return render_template("result_admin.html", initial_images=initial_result_images, pp_images=pp_images,
                                   width=original_width, height=original_height, distance=distances, score=scores,
                                   information=xl[int(best_index) - 1], user=user, error2=error)


    else:
        if (float(distances[0]) > BEFORE_TT_THRESHOLD_2) or (
                            BEFORE_TT_THRESHOLD_1 < float(distances[0]) <= BEFORE_TT_THRESHOLD_2 and float(
                    scores[0]) >= AFTER_PP_THRESHOLD_2):
            text_file.write('Output (found result): \n')
            text_file.write(
                'Top {} scores (cosine similarity) before post-processing: {}, {}, {} \n'.format(3, distances[0],
                                                                                                 distances[1],
                                                                                                 distances[2]))
            text_file.write(
                'Top {} scores (matching point) after post-processing: {}, {}, {} \n'.format(3, scores[0], scores[1],
                                                                                             scores[2]))
            text_file.close()
            return render_template("result_user.html", initial_images=initial_result_images[0], pp_images=pp_images[0],
                                   width=original_width, height=original_height, distance=distances[0], score=scores[0],
                                   information=xl[int(best_index) - 1], user=user)

        elif float(distances[0]) < BEFORE_TT_THRESHOLD_1 or (
                            BEFORE_TT_THRESHOLD_1 < float(distances[0]) <= BEFORE_TT_THRESHOLD_2 and float(
                    scores[0]) < AFTER_PP_THRESHOLD_1):
            error = True
            text_file.write('Output (no result): \n')
            text_file.write(
                'Top {} scores (cosine similarity) before post-processing: {}, {}, {} \n'.format(3, distances[0],
                                                                                                 distances[1],
                                                                                                 distances[2]))
            text_file.write(
                'Top {} scores (matching point) after post-processing: {}, {}, {} \n'.format(3, scores[0], scores[1],
                                                                                             scores[2]))
            text_file.close()
            return render_template("result_user.html", initial_images=initial_result_images[0], pp_images=pp_images[0],
                                   width=original_width, height=original_height, distance=distances[0], score=scores[0],
                                   information=[None], user=user, error1=error)

        else:
            error = True
            text_file.write('Output (uncertain result): \n')
            text_file.write(
                'Top {} scores (cosine similarity) before post-processing: {}, {}, {} \n'.format(3, distances[0],
                                                                                                 distances[1],
                                                                                                 distances[2]))
            text_file.write(
                'Top {} scores (matching point) after post-processing: {}, {}, {} \n'.format(3, scores[0], scores[1],
                                                                                             scores[2]))
            text_file.close()
            return render_template("result_user.html", initial_images=initial_result_images[0], pp_images=pp_images[0],
                                   width=original_width, height=original_height, distance=distances[0], score=scores[0],
                                   information=xl[int(best_index) - 1], user=user, error2=error)


@app.route('/<folder>/<filename>')
def send_image(folder, filename):
    path = os.path.join("users", folder)
    return send_from_directory(path, filename)


if __name__ == "__main__":
    old_images = os.path.join(APP_ROOT, 'images')
    if os.path.isdir(old_images):
        shutil.rmtree(old_images)

    global is_load, current_user_list, admin_directory
    is_load = False
    current_user_list = []
    admin_directory = os.path.join(APP_ROOT, 'admin', 'user_activities')
    if not os.path.exists(admin_directory):
        os.makedirs(admin_directory)
    app.secret_key = os.urandom(12)
    app.run(host='0.0.0.0', port=4555, debug=True)  # 131.159.18.209:4555
