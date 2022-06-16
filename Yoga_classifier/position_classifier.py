import os
from typing_extensions import Self
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from PIL import Image
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


data_dir ="Yoga" 

def data_count(path = "Yoga"):
    '''
    This func take path and counts how many files per folder in the path
    returns a dict with the name of the folder and files in it, if folder is empty
    will not return it.
    '''
    len_files = []
    total_files = 0
    class_name = os.listdir(path)
    for root, dirs, files in os.walk(path):
        if files != []:
            len_files.append(len(files))
            total_files += len(files)
    
    return dict(zip(class_name,len_files))



files_per_class = data_count()
class_names = files_per_class.keys()
total_files = sum(files_per_class.values())
#plt.bar(class_name,files_per_class.values())
#plt.show()

def create_unique_name(class_names, path = "Yoga"):
    '''
    This func create a list of lists, one list per folder in path 
    and in each list store a unique file name 
    '''
    image_files = [[
        os.path.join(path, class_names, x)
        for x in os.listdir(os.path.join(path, class_names)) ]
        for class_names in class_names
        ]
    return image_files
#Here we create a list with names per folder for each image
image_files = create_unique_name(class_names)
image_file_names = []
image_file_labels = []
#Create a list with names of each image
for image_file in image_files :
    image_file_names.extend(image_file)
#Create a list with labels of each image 
image_file_labels = [
    key for key in files_per_class 
    for i in range(files_per_class[key])
]

def data_transformation(image_paths):
    btw_im = [Image.open(image).convert("L") for image in image_paths]
    matrix_representation = [np.array(image)/255 for image in btw_im]
    return matrix_representation

transformed_data = data_transformation(image_paths = image_file_names)

#Create train and test set with stratify on the yoga position
Xtrain, Xtest, Ytrain, Ytest = train_test_split(
    transformed_data,
    image_file_labels,
    test_size = 0.2,
    stratify = image_file_labels
)










pass