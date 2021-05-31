import os
import cv2
import numpy as np
import pandas as pd
import subprocess as sb
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras.backend as K

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, root, target_size=(640,512),batch_size=32,shuffle=True):
        self.rootDir = root
        self.labels = [5, 10, 25, 50, 100]
        self.target_size = target_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.__load_indexing()
        

    def __load_indexing(self):
        self.images_path = self.rootDir
        self.images_list = os.listdir(self.images_path)
        self.len = int(np.floor(len(self.images_list)/self.batch_size))
        self.indexes = np.arange(len(self.images_list))

    def __len__(self):
        return self.len

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.images_list))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        list_images = [self.images_list[k] for k in indexes]

        width,height = self.target_size
    
        X,Y = [],[]

        for image in list_images:
            image_path = os.path.join(self.images_path,image)
            name_split = image.split("_")
            Y.append(self.labels.index(int(name_split[0])))
            image1 = cv2.imread(image_path)
            
            in_img = cv2.resize(image1,self.target_size)
            in_img = np.transpose(in_img,(1,0,2))
            X.append(in_img)

            
        X,Y= np.array(X), np.array(Y)        
        # X = (1./255)*X
        return X,Y

if __name__ == "__main__":
    training_generator = DataGenerator("temp",
            target_size=(640,512),batch_size=2,shuffle=True)
    item = training_generator.__getitem__(5)
    print(item)
