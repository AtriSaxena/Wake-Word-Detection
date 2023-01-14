from tqdm import tqdm
import os 
import sys 
sys.path.append("..")
from utils.util import *
import numpy as np
from tempfile import TemporaryFile

class Data2Features:
    def __init__(self, target_class:str, data_path):
        self.target_class = target_class
        self.data_path = data_path
        self.target_label = 1
        self.non_target_label = 0
        self.features = [] 
        #print(self.data_path)
        #print(filter(os.path.isdir, os.listdir(self.data_path)))
        self.classes = next(os.walk(self.data_path))[1]
        self.classes.remove("_background_noise_") #os.listdir(self.data_path).remove("_background_noise_")
        print(self.classes)
        self.non_target_class = self.classes #['seven','six'] 
        self.label = []
        self.non_target_class.remove(self.target_class) #['seven'] 
        print(f"Following Classes are present:{self.classes}")
    
    def create_features(self):
        self.create_non_target_class()
        self.create_target_class()
        # Balance dataset later 
        self.features = np.asarray(self.features, dtype=np.float32 )
        self.label = np.asarray(self.label, dtype=np.int32)
        print(self.features.shape, self.label.shape)
        X_train, X_test, y_train, y_test = create_train_test_split(self.features, self.label)
        X_train_resample, y_train_resample = make_balanced_class(X_train, y_train)
        return X_train_resample, X_test, y_train_resample, y_test

    def create_non_target_class(self):
        for non_t_class in self.non_target_class:
            print(f"Creating feature for {non_t_class} Non Targeted Class.")
            for file_name in tqdm(os.listdir(os.path.join(self.data_path, non_t_class))):
                lfbe_feature = create_lfbe_feature(file_name=os.path.join(self.data_path, non_t_class, file_name))
                #print(lfbe_feature.shape)
                if lfbe_feature.shape[0] == 76:
                    self.features.append(lfbe_feature)
                    self.label.append(self.non_target_label)
    
    def create_target_class(self):
        print(f"Creating features for target class: {self.target_class}")
        for file_name in tqdm(os.listdir(os.path.join(self.data_path, self.target_class))):
            lfbe_feature = create_lfbe_feature(file_name=os.path.join(self.data_path, self.target_class, file_name))
            if lfbe_feature.shape[0] == 76:
                self.features.append(lfbe_feature)
                self.label.append(self.target_label)

