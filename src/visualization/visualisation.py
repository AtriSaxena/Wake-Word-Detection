import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np 
import random
import cv2 
import os

class VisualizeData: 
    def __init__(self, exp_name):
        self.exp_name = exp_name 
        
    def visualize_train_data(self,X_train: np, y_train:np):
        fig, axs= plt.subplots(nrows=5, ncols = 4, figsize = (8,5), squeeze=True, gridspec_kw={'wspace':0,'hspace':0.25})
        for i, ax in enumerate(axs.ravel()):
            random_index = random.randint(0, len(X_train)) 
            ax.imshow(X_train[random_index], interpolation=None) 
            ax.axis('off')
            if y_train[random_index]==0:
                target_label = "Label: Not Wakeword"
            else:
                target_label = "Label: WakeWord STOP"
            ax.set(title = target_label)
            ax.title.set_fontsize(8)
 
        if not(os.path.exists(self.exp_name)):
            os.mkdir(self.exp_name)
        plt.savefig(self.exp_name / "Training_data.jpg")
        print(f"Training Data Visualization Saved at: {self.exp_name / 'Training_data.jpg'}")
