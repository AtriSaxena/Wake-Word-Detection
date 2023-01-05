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

DATASET_PATH = "B:\Datasets\speech_commands_v0.01"
CLASS_NAME = "stop"

def create_lfbe_feature(file_name):

    samples, sample_rate = librosa.load(file_name)
    lfbe_feature = python_speech_features.base.logfbank(samples, 
                                                        samplerate=sample_rate,
                                                        winlen=0.025,
                                                        winstep=0.01,
                                                        nfilt=64,
                                                        nfft=551)
    print(f"lfbe features:{lfbe_feature.shape}")
    return lfbe_feature[:76][:][:]

def create_train_test_split(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42, shuffle=True)
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

def make_balanced_class(features):
    print(f"Class Distribution before Sampling: {Counter(features[:,1])}")
    over = SMOTE(sampling_strategy=0.1) 
    under = RandomUnderSampler(sampling_strategy=0.5) 
    steps = [('o', over),('u', under)]
    pipeline = Pipeline(steps=steps) 
    X, y = pipeline.fit_resample(features[:,0], features[:,1]) 
    print(f"Class Distribution after Sampling: {Counter(y)}")
    return X, y

def Wave2Spectogram():
    file_name = os.path.join(DATASET_PATH, CLASS_NAME , "0bd689d7_nohash_2.wav")
    samples, sample_rate = librosa.load(file_name)
    print(sample_rate)
    logfbank = python_speech_features.base.logfbank(samples, samplerate=sample_rate, nfilt=64, nfft=551)
    logfbank = logfbank[:76][:][:]
    print(logfbank.shape)
    plt.imshow(logfbank)
    #plt.show()
    plt.savefig("logfbank.png", bbox_inches="tight", pad_inches=0 )
    # fig, ax = plt.subplots(figsize = (5,5))

    # ax.set_axis_off()
    # ax.specgram(samples, Fs=2)
    # fig.savefig("spec.png", bbox_inches="tight", pad_inches=0)
    # plt.close(fig)
    #del sample_rate, ax, fig, samples
Wave2Spectogram()