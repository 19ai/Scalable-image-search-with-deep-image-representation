#!flask/bin/python
# cd in to root folder, then command: source activate ./flask/
import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cuda0,floatX=float32"
import random
import string
from scipy.misc import imread, imsave
from functools import reduce
import re
import sys
import numpy as np
import glob
import pandas as pd
from visualization import visualization
from logistic_regression import logistic_regression

from flask import Flask, request, render_template, send_from_directory, send_file

__author__ = 'ibininja'

app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
DATABASE_FOLDER = 'output/statistics/'

rand_str = lambda n: ''.join([random.choice(string.lowercase + string.digits) for i in xrange(n)])
global images, desc, information, statistics, extra_statistics, extra_extra_statistics, threshold, non_valid_count, file_name


def convert(x):
    statistics_compiler = re.compile('\d+\/')
    return int(re.search(statistics_compiler, x).group()[:-1])


def calculate_accuracy(threshold_1, threshold_2, threshold_3):
    data = pd.read_csv(DATABASE_FOLDER + file_name, header=0, sep=' ', usecols=[0, 1, 2, 3, 6],
                       converters={3: convert}).values
    img_name = data[:, 0]
    cosine_similarities = data[:, 1]
    matching_points = data[:, 2]
    key_points = data[:, 3]
    results = data[:, 4]
    correct_before_pp_condition_1 = np.where(cosine_similarities >= threshold_1)
    correct_before_pp_condition_2 = np.where(((results == 'correct-correct') | (results == 'correct-incorrect')))
    correct_before_pp = np.intersect1d(correct_before_pp_condition_1[0], correct_before_pp_condition_2[0])
    correct_after_pp_condition_1 = np.where(cosine_similarities < threshold_1)
    correct_after_pp_condition_2 = np.where(matching_points >= threshold_2)
    correct_after_pp_condition_3 = np.where(key_points >= threshold_3)
    correct_after_pp_condition_4 = np.where(((results == 'correct-correct') | (results == 'incorrect-correct')))
    correct_after_pp = reduce(np.intersect1d, (correct_after_pp_condition_1[0], correct_after_pp_condition_2[0],
                                               correct_after_pp_condition_3[0], correct_after_pp_condition_4[0]))

    correct_predictions = len(correct_before_pp) + len(correct_after_pp)

    non_valid_condition_1 = np.where(cosine_similarities < threshold_1)
    non_valid_condition_2 = np.where(key_points < threshold_3)
    non_valid = np.intersect1d(non_valid_condition_1[0], non_valid_condition_2[0])
    total_predictions = data.shape[0] - non_valid.shape[0]

    return correct_predictions, total_predictions, non_valid.shape[0]


def calculate_statistics():
    information = []
    f = open(DATABASE_FOLDER + file_name, "r")
    f.next()
    correct_before = np.asarray(
        [float(line.split()[1]) for line in f if len(line.split()) == 7 and line.split()[6][:7] == 'correct'])
    f.close()
    f = open(DATABASE_FOLDER + file_name, "r")
    f.next()
    incorrect_before = np.asarray(
        [float(line.split()[1]) for line in f if len(line.split()) == 7 and line.split()[6][:9] == 'incorrect'])
    f.close()
    f = open(DATABASE_FOLDER + file_name, "r")
    f.next()
    correct_after = np.asarray(
        [float(line.split()[2]) for line in f if len(line.split()) == 7 and line.split()[6][-8:] == '-correct'])
    f.close()
    f = open(DATABASE_FOLDER + file_name, "r")
    f.next()
    incorrect_after = np.asarray(
        [float(line.split()[2]) for line in f if len(line.split()) == 7 and line.split()[6][-9:] == 'incorrect'])

    f.close()

    information.append(
        ['Correct prediction before PP', correct_before.shape[0], np.mean(correct_before), np.std(correct_before),
         np.min(correct_before), np.max(correct_before)])
    information.append(['Incorrect prediction before PP', incorrect_before.shape[0], np.mean(incorrect_before),
                        np.std(incorrect_before),
                        np.min(incorrect_before), np.max(incorrect_before)])
    information.append(
        ['Correct prediction after PP', correct_after.shape[0], np.mean(correct_after), np.std(correct_after),
         np.min(correct_after), np.max(correct_after)])
    information.append(
        ['Incorrect prediction after PP', incorrect_after.shape[0], np.mean(incorrect_after), np.std(incorrect_after),
         np.min(incorrect_after), np.max(incorrect_after)])
    extra_information = []
    acc = float(100 * correct_before.shape[0]) / (correct_before.shape[0] + incorrect_before.shape[0])
    extra_information.append(
        ['Only use result before PP', "{} %".format(acc), logistic_regression(file=file_name, mode='before')])
    acc = float(100 * correct_after.shape[0]) / (correct_after.shape[0] + incorrect_after.shape[0])
    extra_information.append(
        ['Only use result after PP', "{} %".format(acc), logistic_regression(file=file_name, mode='after')])
    extra_extra_information = []
    f = open(DATABASE_FOLDER + file_name, "r")
    f.next()
    labels = [line.split()[6] for line in f]
    f.close()
    extra_extra_information.append(['correct-correct', labels.count('correct-correct')])
    extra_extra_information.append(['incorrect-correct', labels.count('incorrect-correct')])
    extra_extra_information.append(['correct-incorrect', labels.count('correct-incorrect')])
    extra_extra_information.append(['incorrect-incorrect', labels.count('incorrect-incorrect')])

    recom_threshold = []
    recom_threshold.append(extra_information[0][2])
    recom_threshold.append(extra_information[1][2])
    return information, extra_information, extra_extra_information, recom_threshold


@app.route("/")
def index():
    for image_name in glob.iglob(DATABASE_FOLDER + '/*.jpg'):
        os.remove(image_name)

    global images, desc, statistics, extra_statistics, extra_extra_statistics, threshold, file_name
    if len(sys.argv) == 2 and int((sys.argv)[1]) == 1:
        file_name = 'general_output_1.txt'
    else:
        file_name = 'general_output.txt'
    images = []

    statistics, extra_statistics, extra_extra_statistics, threshold = calculate_statistics()
    visualization(file=file_name, mode_1='correct', mode_2='before')
    im = imread(DATABASE_FOLDER + 'correct_before.jpg', mode='RGB')
    new_file_name = rand_str(20)
    imsave(DATABASE_FOLDER + new_file_name + '.jpg', im)
    images.append(new_file_name + '.jpg')

    visualization(file=file_name, mode_1='incorrect', mode_2='before')
    im = imread(DATABASE_FOLDER + 'incorrect_before.jpg', mode='RGB')
    new_file_name = rand_str(20)
    imsave(DATABASE_FOLDER + new_file_name + '.jpg', im)
    images.append(new_file_name + '.jpg')

    logistic_regression(file=file_name, mode='before')
    im = imread(DATABASE_FOLDER + 'LR_before_pp.jpg', mode='RGB')
    new_file_name = rand_str(20)
    imsave(DATABASE_FOLDER + new_file_name + '.jpg', im)
    images.append(new_file_name + '.jpg')

    visualization(file=file_name, mode_1='correct', mode_2='after')
    im = imread(DATABASE_FOLDER + 'correct_after.jpg', mode='RGB')
    new_file_name = rand_str(20)
    imsave(DATABASE_FOLDER + new_file_name + '.jpg', im)
    images.append(new_file_name + '.jpg')

    visualization(file=file_name, mode_1='incorrect', mode_2='after')
    im = imread(DATABASE_FOLDER + 'incorrect_after.jpg', mode='RGB')
    new_file_name = rand_str(20)
    imsave(DATABASE_FOLDER + new_file_name + '.jpg', im)
    images.append(new_file_name + '.jpg')

    logistic_regression(file=file_name, mode='after')
    im = imread(DATABASE_FOLDER + 'LR_after_pp.jpg', mode='RGB')
    new_file_name = rand_str(20)
    imsave(DATABASE_FOLDER + new_file_name + '.jpg', im)
    images.append(new_file_name + '.jpg')

    desc = ['Correct prediction before post-processing', 'Wrong prediction before post-processing',
            'Logistic Regression before post-processing',
            'Correct prediction after post-processing', 'Wrong prediction after post-processing',
            'Logistic Regression after post-processing']

    return render_template("statistics.html", image_name=images, descriptions=desc, first_thres=threshold[0],
                           second_thres=threshold[1],
                           third_thres=5, statistics=statistics, extra_statistics=extra_statistics,
                           extra_extra_statistics=extra_extra_statistics)


@app.route('/calculate', methods=["POST"])
def calculate():
    global images, desc, information, statistics, extra_statistics, extra_extra_statistics, threshold, non_valid_count
    information = []
    x = float(request.form['myRange1'])
    y = float(request.form['myRange2'])
    z = int(request.form['myRange3'])
    correct_predictions, total_predictions, non_valid_count = calculate_accuracy(x, y, z)
    accuracy = float(100 * correct_predictions) / total_predictions

    information.append(['Cosine similarity', x])
    information.append(['Matching points', y])
    information.append(['Minimum key points', z])
    information.append(
        ['Correct/total predictions (Accuracy)', "{}/{} ({} %)".format(correct_predictions, total_predictions,
                                                                       accuracy)])
    information.append(['Non-valid images',non_valid_count])
    return render_template("statistics.html", image_name=images, descriptions=desc, first_thres=x, second_thres=y,
                           third_thres=z, information=information, statistics=statistics,
                           extra_statistics=extra_statistics,
                           extra_extra_statistics=extra_extra_statistics)


@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("output/statistics", filename)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)  # 131.159.18.209:4555
