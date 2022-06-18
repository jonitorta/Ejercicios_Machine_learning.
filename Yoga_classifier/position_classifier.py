from cgi import test
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from PIL import Image
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score


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
    and in each list store a unique file name per file 
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
#Here we chose what class to predict.
target_class = "Tree"
image_class_labels = (np.asarray(image_file_labels) == np.asanyarray(target_class))
#Create train and test set
test_frac = 0.2
Xtrain, Ytrain, Xtest, Ytest = [],[],[],[]
for i in range(total_files):
    rann = np.random.random()
    if rann < test_frac:
        Xtest.append(image_file_names[i])
        Ytest.append(image_class_labels[i])
    else :
        Xtrain.append(image_file_names[i])
        Ytrain.append(image_class_labels[i])

def data_transformation(image_paths):
    '''
    Transform color images to a data frame, each row represent
    an image and the column the pixel with values 0 to 1 
    '''
    btw_im = [Image.open(image).resize((130,130)).convert("L") for image in image_paths]
    array_representation = [np.array(image).ravel()/255 for image in btw_im]
    columns = [ f"pixel {i}" for i in range(130*130)]
    pixel_color_df = pd.DataFrame(data = array_representation, columns=columns)
    return pixel_color_df
    
#Transform data
transformed_data = data_transformation(image_paths = Xtrain)

#Here we use sdg to train the model 
sgd_clf = SGDClassifier(random_state = 1)
#just 3 folds because train set is small (700 samples)
#cross_scores = cross_val_score(sgd_clf, transformed_data, Ytrain, cv=3, scoring="accuracy")
#Here we return the prediction compared to the actual values
#y_train_prediction = cross_val_predict(sgd_clf, transformed_data, Ytrain)
#Create confusion matrix
#conf_matrix = confusion_matrix(Ytrain, y_train_prediction)
#Compute precision, recall and f1 score that is the harmonic mean of precision and recall scores.
#precision_scr = precision_score(Ytrain, y_train_prediction)
#recall_scr = recall_score(Ytrain, y_train_prediction)
#f1_scr = f1_score(Ytrain, y_train_prediction)
#To check the best score 
y_scores = cross_val_predict(sgd_clf, transformed_data, Ytrain,
                                        method = "decision_function")
from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(Ytrain, y_scores)

def plot_precision_recall_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")


plot_precision_recall_threshold(precisions, recalls, thresholds)
plt.show()


pass