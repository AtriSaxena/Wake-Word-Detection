import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np 
import random
import cv2 
import os

class VisualizeData: 
    def __init__(self, exp_name, epochs):
        self.exp_name = exp_name 
        self.epochs = epochs
        
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

    def visualize_training_metrics(self, all_train_loss, all_val_loss, all_train_metrics, all_val_metrics):
        plt.figure(figsize=(5,5))
        plt.plot(np.arange(0,self.epochs,1) ,all_train_loss)
        plt.plot(np.arange(0,self.epochs,1) ,all_val_loss)
        plt.legend(['Training loss','Validation loss'])
        plt.title('Training-Validation Loss Plot')
        plt.savefig(self.exp_name / 'Training-Validation_Loss_Plot.jpg')
        print(f"Loss Visualization Saved at: {self.exp_name / 'Training-Validation_Loss_Plot.jpg'}")
        
        plt.figure(figsize=(5,5))
        plt.plot(np.arange(0,self.epochs,1) ,all_train_metrics['train_accuracy'])
        plt.plot(np.arange(0,self.epochs,1) ,all_val_metrics['val_accuracy'])
        plt.plot(np.arange(0,self.epochs,1) ,all_train_metrics['train_precision'])
        plt.plot(np.arange(0,self.epochs,1) ,all_val_metrics['val_precision'])
        plt.plot(np.arange(0,self.epochs,1) ,all_train_metrics['train_recall'])
        plt.plot(np.arange(0,self.epochs,1) ,all_val_metrics['val_recall'])
        plt.plot(np.arange(0,self.epochs,1) ,all_train_metrics['train_f1score'])
        plt.plot(np.arange(0,self.epochs,1) ,all_val_metrics['val_f1score'])
        plt.legend(['train_accuracy','val_accuracy','train_precision',
                    'val_precision','train_recall','val_recall',
                    'train_f1score','val_f1score'])
        plt.title('Training-Validation Metrics Plot')
        plt.savefig(self.exp_name / 'Training-Validation_Metrics_Plot.jpg')
        print(f"Metrics Visualization Saved at: {self.exp_name / 'Training-Validation_Metrics_Plot.jpg'}")