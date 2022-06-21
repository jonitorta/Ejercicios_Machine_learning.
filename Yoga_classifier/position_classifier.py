import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from sklearn import metrics


data_dir ="YogaPoses" 

def data_structure(path):
    '''
    Returns a dict with folder name and files in folder
    :param path : str path of the main folder
    :return dict : dict with keys as folders in path and values as files per folder and
    the total number of files in main folder
    '''

    len_files = []
    total_files = 0
    folders_name = os.listdir(path)
    for root, dirs, files in os.walk(path):
        if files != []:
            len_files.append(len(files))
            total_files += len(files)
    
    return dict(zip(folders_name,len_files)), total_files


files_per_folder, total_files = data_structure(data_dir)

def create_unique_name(path):
    '''
    Create unique names for each file in main folder.
    :param path : str path of the main folder
    :return list : list with unique name per file
    '''
    folders_name = os.listdir(path)
    image_names = [
        os.path.join(path,folder_name,x)
        for folder_name in folders_name
        for x in os.listdir(os.path.join(path,folder_name))
    ]
    return image_names

image_names = create_unique_name(data_dir)
#Create a list with the class of each file
images_list = [
    key for key in files_per_folder  
    for i in range(files_per_folder[key])
]

#Lets see how many images each class has.
#print(files_per_folder)
#Lets print an image per category.
def plot_images(path,category):
    '''
    Print 3 images of given category
    :param path :str path of the main folder
    :param category :str name of category
    '''
    images_path = os.path.join(path,category)
    images_name =   [ os.path.join(images_path,name) for name in os.listdir(images_path)[0:3] ]
    
    for idx, name in enumerate(images_name):
        image = mpimg.imread(name)
        plt.figure()
        plt.imshow(image)
    plt.show()

#Lets see some images.
#plot_images(data_dir, "Tree")

#Create a test train sets.
test_frac = 0.2
Xtrain, Ytrain, Xtest, Ytest = [],[],[],[]
for i in range(total_files):
    rann = np.random.random()
    if rann < test_frac:
        Xtest.append(image_names[i])
        Ytest.append(images_list[i])
    else :
        Xtrain.append(image_names[i])
        Ytrain.append(images_list[i])

def data_transformation(images_path):
    '''
    Transform images rescaling and to gray color.
    :param images_path: list with image paths.
    :return df: df with pixel intensity from 0 to 1 in columns and image in rows.
    '''
    resize_gray_images = [Image.open(image).resize((150,150)).convert("L")
        for image in images_path
    ]
    
    array_representation = [
        np.array(transformed_image).ravel()/255
        for transformed_image in resize_gray_images
    ]

    column_names = [f"pixel {i}" for i in range(150*150)]

    pixel_color_df = pd.DataFrame(data = array_representation, columns= column_names)

    return pixel_color_df
#Transform data
pixel_df = data_transformation(Xtrain)
#Chose a target class
target_class = "Tree"
#Create a list false when is not target class true when is
is_target = (np.asarray(Ytrain) == target_class )
#Create a classifier
from sklearn.linear_model import SGDClassifier
sdg_clf = SGDClassifier(random_state=1)

from sklearn.model_selection import cross_val_score
accuracy = cross_val_score(sdg_clf, pixel_df, is_target, scoring = "accuracy", cv=3)

from sklearn.model_selection import cross_val_predict
predictions = cross_val_predict(sdg_clf, pixel_df, is_target, cv=3)

from sklearn.metrics import confusion_matrix
cnf_mtx = confusion_matrix(is_target, predictions)

#from sklearn.metrics import precision_score, recall_score, , f1_score

class_scores = cross_val_predict(sdg_clf, pixel_df, is_target, cv=3, method = "decision_function")     

from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score

precisions, recalls, thresholds = precision_recall_curve(is_target,class_scores)

def plot_precisions_recalls_vs_threshold(precisions,recalls,thresholds):
    '''
    Plot precision and recall vs threshold
    '''
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.show()

plot_precisions_recalls_vs_threshold(precisions,recalls,thresholds)

false_positive_rate, true_positive_rate, thresholds = roc_curve(is_target,class_scores)

def plot_roc_curve(fpr, tpr, label=None):
    '''
    Plot roc curve
    '''
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.show()

plot_roc_curve(false_positive_rate, true_positive_rate, thresholds)

from joblib import dump

def store_metrics(*args, name = "Metrics"):
    '''
    Save dict in pkl format with key as variable name and 
    value the value assigned to the variable, file is named with
    name param.
    :param **kargs: array, matrix or numeric value to store. 
    :param name : str name of file to save
    '''
    def return_variable_name(variable_name):
        '''
        Returns variable name as a string
        '''
        return f"{variable_name=}".split("=")[0]


    names = [return_variable_name(variable) for variable in args]
    values = [values for values in args]

    to_save = pd.DataFrame(dict(zip(names, values)))
    
    return to_save
    #dump(to_save, name)

to_save = store_metrics(accuracy,
             predictions,
             cnf_mtx,
             name="Metrics")

open_metrics = pd.read_pickle("Metrics")

    



pass