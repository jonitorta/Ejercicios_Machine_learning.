import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


data_dir ="Yoga" 

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
    




pass