# Scalable Image Retrieval
**Project Topic** - Scalable Image Search with Deep Image Representation

## Introduction

Here is the implementation for my master thesis, which I completed in February 2018, in the topic of Image Retrieval. An image retrieval model is a set of algorithms for browsing, searching and retrieving images from a large database of images. Generally, given an input image from a user, an image retrieval model needs to compare that image with all images in a particular database, and returns the most similar image to user.

Image retrieval models in realistic scenarios have to deal with two main challenges: Large-scale datasets of unlabeled images and response time. Indeed, with the growth of Internet as well as widespread of media, images could be now easily taken from numerous sources, leading to huge collections of image. When image datasets could be updated everyday, it is neither efficient nor scalable to re-train a model whenever
new images are added. Convolutional Neural Networks have created new perspectives for Computer Vision and have been widely appreciated as an effective way to extract features for image retrieval tasks.

In this master thesis, based on the well-known pre-trained VGG model, which are trained over 1000 classes, we propose a way to extract convolutional features based on semantic information predicted in the target image, without the need for label of image. By using Class Activation Maps (CAMs) and linear as well as non-linear operations including weighting, pooling and aggregation, we could encode every image by a fix-size vector, used for image similarity measurement. In the post-processing stage, we address the second challenge mentioned above: response time. By performing feature extraction directly on images and then feature matching, we could fine-tune initial result
with much better accuracy, and less than a second for computation, which is to the best of our knowledge, faster than all introduced Image Retrieval steps. Furthermore, our model has an advantage of well-scalable to huge datasets.

We evaluate our model on an image collection of European fine arts, namely "Web Gallery of Art", including 43690 image representing painting, sculptures, etc. We obtain an outstanding precision of 99.98% with less than 2 seconds for responding a query image.

## Pre-requisites

The training phase of this project is run on GPU, with tensorflow backend. Therefore please make sure that your machine has at least 6GB GPU
Please install the following libraries/frameworks to run our code:

- **Tensorflow** 
- **Numpy** 
- **OpenCV** 
- **Keras** 

If you want to run the web application, please install the following framework:

- **flask** 
- **flask_migrate** 
- **flask_script** 
- **flask_mail** 
- **flask_user** 

We highly recommend you to create a virtual environment by using Anaconda. If you follow that suggestion, you could run the following 3 commands on terminal sequentially to complete the set up:

- **./installation_1.sh**: Create a virtual environment
- **source activate vir_env**: Activate the created environment 
- **./installation_2.sh**: Install necessary stuff

## Dataset

The dataset used for our experiments names **Web Gallery of Art** (https://www.wga.hu/). It is an image collection of European fine arts, including paintings and sculptures, ... from the 8th to 19th centuries. Some sample images are shown in Figure 4.1. In total, there
are 43690 images in this dataset. Additionally, we also have a database containing information for each image such as title, date, technique, form and so on which could be used to provide further information with final result (but it does not contribute anything to the model’s performance). The database file could be dow

## Preparation

Firstly, you need to put 3 folders **database**, **dataset** and **output** in the following link https://drive.google.com/open?id=1e5pOxnnULjmCh8QSXhIT2gGuwZhbKE2R to the current directory. Since the dataset we collected is private, we can not publish it, but you could download all images we collected from the link provided in the **Dataset** section. You could split and store the whole dataset in sub-folders (In our case, we create 15 sub-folders, with 3000 images for each folder, except from the last one). Then you need to run the script create_list.py to create 2 lists of horizontal and vertical images, for faster training phase. 

## Training

For training the model and pre-compute post-processing data, please run the following commands:
- **python train.py**
- **python pre_compute_descriptor.py**

## Test

For testing the model with query images inside a folder, please run the following command:
- **python test_with_statistics.py -n path/to/folder/**

## Web application
For start application's server, please run the following command:
- **python app.py**

## Web application's statistics
For visualizing the statistics of the web application, you could run the following command:
- **python app_statistics.py**
