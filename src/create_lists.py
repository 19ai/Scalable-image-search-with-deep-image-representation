import os
from scipy.misc import imread

def create_lists(folder_path, dataset_name):
     horizontal_file = open("../datasets/lists/list_"+dataset_name+"_horizontal.txt", "w")
     vertical_file = open("../datasets/lists/list_"+dataset_name+"_vertical.txt", "w")
     count = 0
     for subdir, dirs, files in os.walk(folder_path):
          for file in files:
               filename = subdir + os.sep + file
               if filename.endswith(".jpg"):
                    count += 1
                    print (filename)
                    image = imread(filename)
                    if image.shape[0] <= image.shape[1]:
                         horizontal_file.write(filename[2:]+'\n')
                    else:
                         vertical_file.write(filename[2:]+'\n')
     horizontal_file.close()
     vertical_file.close()
     print count

if __name__ == '__main__':
     create_lists(folder_path = '../datasets/images/painting', dataset_name = 'painting')