import librosa
import os 
import matplotlib.pyplot as plt 
import python_speech_features
from sklearn.model_selection import train_test_split
import numpy as np 
from imblearn.over_sampling import SMOTE 
from imblearn.under_sampling import RandomUnderSampler 
from imblearn.pipeline import Pipeline 
from collections import Counter
import pandas as pd

def create_lfbe_feature(file_name):

    samples, sample_rate = librosa.load(file_name)
    lfbe_feature = python_speech_features.base.logfbank(samples, 
                                                        samplerate=sample_rate,
                                                        winlen=0.025,
                                                        winstep=0.01,
                                                        nfilt=64,
                                                        nfft=551)
    return lfbe_feature[:76][:][:]

def create_train_test_split(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42, shuffle=True)
    return X_train, X_test, y_train, y_test

def load_features(file_name = "wakeword_features.npz"):
    """
    For loading the npz features file.
    """
    features = np.load(file_name, allow_pickle=True)
    X_train = features['X_train'] 
    y_train = features['y_train']
    X_test = features['X_test'] 
    y_test = features['y_test']
    return X_train, y_train, X_test, y_test

def make_balanced_class(X_train, y_train):
    print(f"Class Distribution before Sampling: {Counter(y_train)}")
    sampling_length = Counter(y_train)[1] + 2000
    over = SMOTE(sampling_strategy={1:sampling_length}) 
    under = RandomUnderSampler(sampling_strategy={0:sampling_length}) 
    steps = [('u', under),('o', over)]
    pipeline = Pipeline(steps=steps) 

    train_rows=len(X_train)
    X_smote = X_train.reshape(train_rows, -1)
    X_train_resample, y_train_resample = pipeline.fit_resample(X_smote, y_train) 

    print(f"Class Distribution after Sampling: {Counter(y_train_resample)}")
    X_train_resample = X_train_resample.reshape(-1, 76,64)
    
    return X_train_resample, y_train_resample

# def Wave2Spectogram():
#     file_name = os.path.join(DATASET_PATH, CLASS_NAME , "0bd689d7_nohash_2.wav")
#     samples, sample_rate = librosa.load(file_name)
#     logfbank = python_speech_features.base.logfbank(samples, samplerate=sample_rate, nfilt=64, nfft=551)
#     logfbank = logfbank[:76][:][:]
#     plt.imshow(logfbank)
#     plt.savefig("logfbank.png", bbox_inches="tight", pad_inches=0 )