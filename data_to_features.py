from tqdm import tqdm
import os 
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
        self.classes = os.listdir(self.data_path).remove("_background_noise_")
        self.non_target_class = self.classes.remove(self.target_class) #['seven'] 
        print(f"Following Classes are present:{os.listdir(self.data_path)}")
    
    def create_features(self):
        self.create_non_target_class()
        self.create_target_class()
        # Balance dataset later 
        self.features = np.asarray(self.features, dtype=object )
        X, y = make_balanced_class(features=self.features)
        return create_train_test_split(X,y)

    def create_non_target_class(self):
        for non_t_class in self.non_target_class:
            print(f"Creating feature for {non_t_class} Non Targeted Class.")
            for file_name in tqdm(os.listdir(os.path.join(self.data_path, non_t_class))):
                lfbe_feature = create_lfbe_feature(file_name=os.path.join(self.data_path, non_t_class, file_name))
                if lfbe_feature.shape[0] == 76:
                    self.features.append((lfbe_feature, self.non_target_label))
    
    def create_target_class(self):
        print(f"Creating features for target class: {self.target_class}")
        for file_name in tqdm(os.listdir(os.path.join(self.data_path, self.target_class))):
            lfbe_feature = create_lfbe_feature(file_name=os.path.join(self.data_path, self.target_class, file_name))
            if lfbe_feature.shape[0] == 76:
                self.features.append((lfbe_feature, self.target_label))


DATASET_PATH = "B:\Datasets\speech_commands_v0.01"
TARGET_CLASS_NAME = "stop"
#NON_TARGET_CLASS_NAME = ["seven"]

data2features = Data2Features(target_class=TARGET_CLASS_NAME, 
                                data_path=DATASET_PATH)
X_train, X_test, y_train, y_test = data2features.create_features()
print(X_train.shape, y_train.shape)
np.savez("wakeword_features", X_train = X_train, 
                            X_test = X_test,
                            y_train = y_train,
                            y_test = y_test)
